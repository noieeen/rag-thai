# from fastapi import File, UploadFile, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uvicorn
# from typing import List, Optional
# import asyncio
# import os
# from pathlib import Path
# import tempfile
# import logging
# from contextlib import asynccontextmanager

# # Import our custom modules
# from services.ocr_service import OCRService
# from services.text_processor import TextProcessor
# from services.embedding_service import EmbeddingService
# from services.vector_store import VectorStoreService
# from models.schemas import (
#     DocumentResponse, 
#     SearchRequest, 
#     SearchResponse,
#     ProcessingStatus,
#     DocumentChunk
# )
# from core.config import settings
# from core.database import get_connection
# from utils.file_utils import save_upload_file, cleanup_temp_file
# from utils.validators import validate_file_type

# # Store for tracking processing status
# processing_status = {}

# async def process_document_background(
#     doc_id: str,
#     file_path: str,
#     filename: str,
#     extract_text: bool,
#     chunk_size: int,
#     overlap: int
# ):
#     """Background task for document processing"""
#     print(f"Processing document {doc_id} at {file_path}")
#     try:
#         # Update status
#         processing_status[doc_id].status = "processing"
#         processing_status[doc_id].progress = 10
#         processing_status[doc_id].message = "Extracting text from document..."
        
#         # Extract text using OCR
#         if extract_text:
#             print(f"Extracting text from file | process_document_background: {file_path}")
#             extracted_text = await ocr_service.extract_text(file_path)
#         else:
#             # For text files, read directly
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 extracted_text = f.read()
        
#         processing_status[doc_id].progress = 30
#         processing_status[doc_id].message = "Processing Thai text..."
        
#         # Process Thai text
#         # processed_text = await text_processor.process_text(extracted_text)
#         processed_text = text_processor.process_text(extracted_text)
        
#         processing_status[doc_id].progress = 50
#         processing_status[doc_id].message = "Creating text chunks..."
        
#         # Create chunks
#         chunks = await text_processor.create_chunks(
#             processed_text, 
#             chunk_size=chunk_size, 
#             overlap=overlap
#         )
        
#         processing_status[doc_id].progress = 70
#         processing_status[doc_id].message = "Generating embeddings..."
        
#         # Generate embeddings for chunks
#         chunk_embeddings = []
#         for i, chunk in enumerate(chunks):
#             embedding = await embedding_service.generate_embedding(chunk.text)
#             chunk.embedding = embedding
#             chunk_embeddings.append(chunk)
            
#             # Update progress
#             progress = 70 + (20 * (i + 1) / len(chunks))
#             processing_status[doc_id].progress = int(progress)
        
#         processing_status[doc_id].progress = 90
#         processing_status[doc_id].message = "Storing in vector database..."
        
#         # Store in vector database
#         await vector_store.store_document(doc_id, filename, chunk_embeddings)
        
#         # Complete processing
#         processing_status[doc_id].status = "completed"
#         processing_status[doc_id].progress = 100
#         processing_status[doc_id].message = "Document processed successfully"
        
#         logger.info(f"Document {doc_id} processed successfully")
        
#     except Exception as e:
#         processing_status[doc_id].status = "failed"
#         processing_status[doc_id].message = f"Processing failed: {str(e)}"
#         logger.error(f"Document processing failed for {doc_id}: {e}")
        
#     finally:
#         # Clean up temporary file
#         cleanup_temp_file(file_path)