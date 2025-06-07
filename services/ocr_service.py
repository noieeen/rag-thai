import easyocr
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import logging
from typing import List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from pythainlp.spell import correct

from core.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    """Service for OCR text extraction from images and PDFs"""
    
    def __init__(self):
        self.reader = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _initialize_reader(self):
        """Initialize EasyOCR reader (lazy loading)"""
        if self.reader is None:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(
                settings.OCR_LANGUAGES,
                gpu=settings.OCR_GPU
            )
            logger.info("EasyOCR reader initialized successfully")
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from file (PDF or image)"""
        
        print(f"Extracting text from file | extract_text: {file_path}")
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return await self._extract_from_pdf(str(file_path))
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return await self._extract_from_image(str(file_path))
        elif file_extension == '.txt':
            return await self._read_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    async def _extract_from_pdf(self, pdf_path: str) -> str:
        print(f"Extracting text from PDF | _extract_from_pdf: {pdf_path}")
        """Extract text from PDF file"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        def _process_pdf():
            extracted_text = []
            
            print(f"Processing PDF | _process_pdf: {pdf_path}")
            # Try to extract text directly first (for text-based PDFs)
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        extracted_text.append(text)
                    else:
                        # If no text found, use OCR on page image
                        logger.info(f"No text found on page {page_num + 1}, using OCR...")
                        ocr_text = self._ocr_pdf_page(page)
                        if ocr_text:
                            extracted_text.append(ocr_text)
                
                doc.close()
            
                print(f"Extracted {len(extracted_text)} pages of text from PDF")
                
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                # Fallback to full OCR processing
                return self._ocr_entire_pdf(pdf_path)
            
            return '\n\n'.join(extracted_text)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _process_pdf
        )
    
    def _ocr_pdf_page(self, page) -> str:
        """Perform OCR on a single PDF page"""
        # Convert page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to numpy array for EasyOCR
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return self._perform_ocr(img)
    
    def _ocr_entire_pdf(self, pdf_path: str) -> str:
        """Perform OCR on entire PDF by converting to images"""
        self._initialize_reader()
        
        extracted_text = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = self._ocr_pdf_page(page)
            if text:
                extracted_text.append(text)
        
        doc.close()
        return '\n\n'.join(extracted_text)
    
    async def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image file"""
        logger.info(f"Extracting text from image: {image_path}")
        
        def _process_image():
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image for better OCR
            img = self._preprocess_image(img)
            
            return self._perform_ocr(img)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _process_image
        )
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (1, 1), 0)
        
        # Apply threshold to get better contrast
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.medianBlur(img, 3)
        
        return img
    
    def _perform_ocr(self, img: np.ndarray) -> str:
        """Perform OCR on preprocessed image"""
        self._initialize_reader()
        
        try:
            # Perform OCR
            results = self.reader.readtext(img, detail=1)
            
            # Extract text with confidence filtering
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter out low-confidence results
                    extracted_text.append(text)
            
            raw_text = ' '.join(extracted_text)
            # Clean up text
            cleaned_text = self.correct_thai_text(raw_text)
            return cleaned_text
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""
    
    async def _read_text_file(self, file_path: str) -> str:
        """Read text from plain text file"""
        logger.info(f"Reading text file: {file_path}")
        
        def _read_file():
            encodings = ['utf-8', 'utf-16', 'cp874', 'latin1']  # cp874 for Thai Windows encoding
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading file with {encoding}: {e}")
                    continue
            
            raise ValueError(f"Could not decode file {file_path} with any supported encoding")
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _read_file
        )
    
    async def extract_with_confidence(self, file_path: str) -> Tuple[str, List[float]]:
        """Extract text with confidence scores"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            def _process_with_confidence():
                self._initialize_reader()
                img = cv2.imread(file_path)
                img = self._preprocess_image(img)
                
                results = self.reader.readtext(img)
                
                texts = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    texts.append(text)
                    confidences.append(confidence)
                
                return ' '.join(texts), confidences
            
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, _process_with_confidence
            )
        else:
            # For PDF and text files, return text with dummy confidence
            text = await self.extract_text(file_path)
            print("Extracted Text:")
            print(text)
            return text, [1.0]  # Assume perfect confidence for non-OCR text
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        self.reader = None
    
    def correct_thai_text(self, text: str) -> str:
        """
        Correct Thai text using pythainlp spell correction.
        This function uses the pythainlp library to correct common spelling mistakes in Thai text.

        Args:
            text (str): The input Thai text to be corrected.
        Returns:
            str: The corrected Thai text.
        -----------
        Example:
            >>> ocr_service = OCRService()
            >>> corrected_text = ocr_service.correct_thai_text("สวัสดีครับ")
            >>> print(corrected_text)
            "สวัสดีครับ"
        -----------
        """
        
        print(f"Correcting Thai text: {text} | correct_thai_text")
        tokens = text.split()
        return ' '.join([correct(token) for token in tokens])