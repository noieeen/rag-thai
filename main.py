from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import asyncio
import os
from pathlib import Path
import tempfile
import numpy as np

import logging
from contextlib import asynccontextmanager

# Import our custom modules
from services.ocr_service import OCRService
from services.text_processor import TextProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStoreService
from models.schemas import (
    DocumentResponse, 
    SearchRequest, 
    SearchResponse,
    ProcessingStatus,
    DocumentChunk
)
from core.config import settings
from core.database import get_connection
# from core.chromadb import get_connection
from utils.file_utils import save_upload_file, cleanup_temp_file
from utils.validators import validate_file_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
ocr_service = None
text_processor = None
embedding_service = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ocr_service, text_processor, embedding_service, vector_store
    
    logger.info("Initializing services...")
    
    try:
        # Initialize services
        ocr_service = OCRService()
        text_processor = TextProcessor()
        embedding_service = EmbeddingService()
        vector_store = VectorStoreService()
        
        # Initialize embedding model
        await embedding_service.initialize()
        
        # Initialize vector store
        await vector_store.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    if vector_store:
        await vector_store.close()

app = FastAPI(
    title="Thai RAG Document Assistant API",
    description="API for processing Thai documents with OCR, text processing, and RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for tracking processing status
processing_status = {}

@app.get("/")
async def root():
    return {"message": "Thai RAG Document Assistant API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "ocr": ocr_service is not None,
            "text_processor": text_processor is not None,
            "embedding": embedding_service is not None,
            "vector_store": vector_store is not None
        }
    }

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_text: bool = True,
    chunk_size: int = 500,
    overlap: int = 50
):
    """Upload and process a document (PDF, image, or text file)"""
    
    # contents = await file.read()
    # print(f"Received file: {file.filename}, size: {len(contents)} bytes, type: {file.content_type}")
    
    # Validate file
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: PDF, PNG, JPG, JPEG, TXT"
        )
    
    # Generate document ID
    doc_id = f"doc_{hash(file.filename + str(asyncio.get_event_loop().time()))}"
    
    # Initialize processing status
    processing_status[doc_id] = ProcessingStatus(
        document_id=doc_id,
        status="uploading",
        progress=0,
        message="File uploaded, starting processing..."
    )
    
    try:  

        # Save uploaded file temporarily
        temp_file_path = await save_upload_file(file,"uploads/" + file.filename)
        
        print(f"\nDocument {doc_id} processing before started in background")

        # Start background processing
        background_tasks.add_task(
            process_document_background,
            doc_id,
            temp_file_path,
            file.filename,
            extract_text,
            chunk_size,
            overlap
        )
        
        
        print(f"\nDocument {doc_id} processing started in background")

        
        return DocumentResponse(
            document_id=doc_id,
            filename=file.filename,
            status="processing",
            message="Document uploaded successfully. Processing started."
        )
        
    except Exception as e:
        processing_status[doc_id] = ProcessingStatus(
            document_id=doc_id,
            status="failed",
            progress=0,
            message=f"Upload failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_background(
    doc_id: str,
    file_path: str,
    filename: str,
    extract_text: bool,
    chunk_size: int,
    overlap: int
):
    """Background task for document processing"""
    print(f"Processing document {doc_id} at {file_path}")
    try:
        # Update status
        processing_status[doc_id].status = "processing"
        processing_status[doc_id].progress = 10
        processing_status[doc_id].message = "Extracting text from document..."
        
        logger.info(f"Processing document {doc_id} | progress: {processing_status[doc_id].progress}%")
        # Extract text using OCR
        if extract_text:
            print(f"Extracting text from file | process_document_background: {file_path}")
            extracted_text = await ocr_service.extract_text(file_path)
        else:
            # For text files, read directly
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        
        processing_status[doc_id].progress = 30
        processing_status[doc_id].message = "Processing Thai text..."
        
        logger.info(f"Processing document {doc_id} | progress: {processing_status[doc_id].progress}%")
        # Process Thai text
        # processed_text = await text_processor.process_text(extracted_text)
        processed_text = text_processor.process_text(extracted_text)
        
        processing_status[doc_id].progress = 50
        processing_status[doc_id].message = "Creating text chunks..."
        
        logger.info(f"Processing document {doc_id} | progress: {processing_status[doc_id].progress}%")
        # Create chunks
        chunks = await text_processor.create_chunks(
            processed_text, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        processing_status[doc_id].progress = 70
        processing_status[doc_id].message = "Generating embeddings..."
        
        logger.info(f"Processing document {doc_id} | progress: {processing_status[doc_id].progress}%")
        # Generate embeddings for chunks
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk.text}")
            embedding = await embedding_service.generate_embedding(chunk.text)
            chunk.embedding = embedding
            chunk_embeddings.append(chunk)
            
            # Update progress
            progress = 70 + (20 * (i + 1) / len(chunks))
            processing_status[doc_id].progress = int(progress)
        
        processing_status[doc_id].progress = 90
        processing_status[doc_id].message = "Storing in vector database..."
        
        # Store in vector database
        await vector_store.store_document(doc_id, filename, chunk_embeddings)
        
        # Complete processing
        processing_status[doc_id].status = "completed"
        processing_status[doc_id].progress = 100
        processing_status[doc_id].message = "Document processed successfully"
        
        logger.info(f"Processing document {doc_id} | progress: {processing_status[doc_id].progress}%")
        
        logger.info(f"Document {doc_id} processed successfully")
        
    except Exception as e:
        processing_status[doc_id].status = "failed"
        processing_status[doc_id].message = f"Processing failed: {str(e)}"
        logger.error(f"Document processing failed for {doc_id}: {e}")
        
    finally:
        # Clean up temporary file
        cleanup_temp_file(file_path)

@app.get("/documents/{doc_id}/status", response_model=ProcessingStatus)
async def get_processing_status(doc_id: str):
    """Get document processing status"""
    
    if doc_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_status[doc_id]

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search through processed documents"""
    
    try:
        # Process the query
        processed_query = await text_processor.process_query(request.query)
        
        logger.info(f"Processed query: {processed_query}")
        
        # Generate query embedding
        # Ensure the input is a single string for the embedding model
        #query_embedding = await embedding_service.generate_embedding(" ".join(processed_query))
        
        ## Test
        embeddings = embedding_service.embed_texts(processed_query)
        query_embedding = np.mean(embeddings, axis=0)
        
        logger.info(f"Generated embedding for query: {query_embedding[:10]}... (truncated)")
        
        logger.info(f"Searching in vector store for query: {request}")
        # Search in vector store
        results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=request.limit,
            threshold=request.threshold,
            filter_doc_ids=request.document_ids
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                {
                    "document_id": meta["doc_id"],
                    "chunk_id": f"{meta['doc_id']}_{meta['start']}_{meta['end']}",
                    "text": meta.get("text", ""),
                    "score": sim
                }
                for meta, sim in results
            ],
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    
    try:
        # documents = await vector_store.list_documents()
        documents = vector_store.list_documents()
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks from vector store"""
    
    try:
        await vector_store.delete_document(doc_id)
        
        # Remove from processing status if exists
        if doc_id in processing_status:
            del processing_status[doc_id]
        
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text/process")
async def process_text_endpoint(text: str):
    """Process Thai text without storing (for testing)"""
    
    try:
        processed = await text_processor.process_text(text)
        chunks = await text_processor.create_chunks(processed)
        
        return {
            "original_text": text,
            "processed_text": processed,
            "chunks": [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
        }
        
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/generate")
async def generate_embedding_endpoint(text: str):
    """Generate embedding for text (for testing)"""
    
    try:
        embedding = await embedding_service.generate_embedding(text)
        
        return {
            "text": text,
            "embedding_dimension": len(embedding),
            "embedding": embedding.tolist()
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )