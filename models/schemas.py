from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True

class DocumentResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    created_at: Optional[datetime] = None

class ProcessingStatus(BaseModel):
    """Model for tracking document processing status"""
    document_id: str
    status: str  # uploading, processing, completed, failed
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SearchRequest(BaseModel):
    """Request model for document search"""
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = None
    
class SearchResult(BaseModel):
    """Model for individual search result"""
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: Optional[float] = None

class TextProcessRequest(BaseModel):
    """Request model for text processing"""
    text: str = Field(min_length=1)
    normalize: bool = True
    remove_stopwords: bool = True
    tokenize: bool = True

class TextProcessResponse(BaseModel):
    """Response model for text processing"""
    original_text: str
    processed_text: str
    tokens: Optional[List[str]] = None
    language_detected: Optional[str] = None
    word_count: int
    char_count: int

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    text: str = Field(min_length=1)
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    text: str
    embedding: List[float]
    dimension: int
    model: str

class DocumentInfo(BaseModel):
    """Model for document information"""
    document_id: str
    filename: str
    status: str
    chunk_count: int
    created_at: datetime
    file_size: Optional[int] = None
    language: Optional[str] = None

class ChunkRequest(BaseModel):
    """Request model for text chunking"""
    text: str = Field(min_length=1)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    
class ChunkResponse(BaseModel):
    """Response model for text chunking"""
    original_text: str
    chunks: List[Dict[str, Any]]
    total_chunks: int
    average_chunk_size: float

class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    languages: List[str] = ["th", "en"]
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class OCRResponse(BaseModel):
    """Response model for OCR results"""
    extracted_text: str
    confidence_scores: List[float]
    detected_languages: List[str]
    processing_time: float

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    services: Dict[str, bool]
    timestamp: datetime
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime