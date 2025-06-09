import easyocr
import cv2
import numpy as np
from PIL import Image
import pymupdf
import logging
from typing import List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from exceptiongroup import catch
from pythainlp.spell import correct
import asyncio

from core.config import settings

from functools import lru_cache, partial  # Import lru_cache and partial

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR text extraction from images and PDFs"""

    def __init__(self):
        self.reader = None
        self.executor = ThreadPoolExecutor(max_workers=8)

    def _initialize_reader(self):
        """Initialize EasyOCR reader (lazy loading)"""
        if self.reader is None:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(
                settings.OCR_LANGUAGES,
                gpu=settings.OCR_GPU
            )
            logger.info("EasyOCR reader initialized successfully")

    @lru_cache(maxsize=2048)  # Cache up to 2048 unique word corrections
    def _cached_correct_word(self, word: str) -> str:
        """Helper function to cache results of pythainlp.spell.correct."""
        try:
            return correct(word)
        except Exception as e:
            logger.warning(f"Error during spell correction for '{word}': {e}. Returning original word.")
            return word

    def correct_thai_text(self, text: str) -> str:
        """
        Corrects Thai text using pythainlp spell correction with caching.
        This function is optimized by caching previously corrected words.
        """
        if not text:
            return ""

        # Simple split by whitespace; for more complex scenarios, consider pythainlp.tokenize.word_tokenize
        tokens = text.split()

        # Apply cached correction to each token
        corrected_tokens = [self._cached_correct_word(token) for token in tokens]

        return ' '.join(corrected_tokens)

    async def extract_text(self, file_path: str) -> str:
        """Extract text from file (PDF or image)"""

        logger.info(f"Starting text extraction for: {file_path}")
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

        def _process_pdf_sync():
            extracted_text_parts = []
            pages_to_ocr_images = []

            print(f"Processing PDF | _process_pdf: {pdf_path}")
            # Try to extract text directly first (for text-based PDFs)
            try:
                doc = pymupdf.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():
                        extracted_text_parts.append(text)
                    else:
                        # If no text found, use OCR on page image
                        logger.info(f"No text found on page {page_num + 1}, using OCR...")
                        # ocr_text = self._ocr_pdf_page(page)
                        # if ocr_text:
                        pages_to_ocr_images.append(page)

                doc.close()

                # Perform OCR on pages that didn't yield text, in parallel
                # if pages_to_ocr_images:
                #     logger.info(f"Performing parallel OCR on {len(pages_to_ocr_images)}")
                #     # Use partial to pass the instance method to map
                #     ocr_results = list(self.executor.map(
                #         partial(self._ocr_pdf_page_sync, self),  # Pass self for instance method
                #         pages_to_ocr_images
                #     ))
                #
                #     for ocr_text in ocr_results:
                #         if ocr_text:
                #             extracted_text_parts.append(ocr_text)
                # Perform OCR on pages that didn't yield text, in parallel
                if pages_to_ocr_images:
                    logger.info(f"Performing OCR on {len(pages_to_ocr_images)} image-based PDF pages in parallel.")

                    # Use self.executor to run OCR for each page concurrently
                    # partial is used to pass arguments to _ocr_pdf_page in the map function
                    ocr_results = list(self.executor.map(self._ocr_pdf_page_sync, pages_to_ocr_images))

                    for ocr_text in ocr_results:
                        if ocr_text:
                            extracted_text_parts.append(ocr_text)

                logger.info(f"Finish processing PDF: {pdf_path}")
                return '\n\n'.join(extracted_text_parts)


            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}. Attempting full image-based OCR fallback.")
                # Fallback to full image-based OCR if initial processing fails
                return self._ocr_entire_pdf_sync(pdf_path)

        return await asyncio.to_thread(_process_pdf_sync)

        # return await asyncio.get_event_loop().run_in_executor(
        #     self.executor, _process_pdf_sync()
        # )

    # Static method to allow easy use with executor.map without binding issues
    @staticmethod
    def _ocr_pdf_page_sync(instance: 'OCRService', page) -> str:
        """
        Synchronously perform OCR on a single PDF page image.
        Takes an instance of OCRService to access its methods/attributes.
        """
        try:
            # Render page to a high-resolution pixmap for better OCR
            mat = pymupdf.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to numpy array via bytes
            img_data = pix.tobytes("png")
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"Failed to decode image from PDF page: {page.number}")
                return ""

            return instance._perform_ocr_sync(img)
        except Exception as e:
            logger.error(f"Error during OCR of PDF page {page.number}: {e}")
            return ""

    def _ocr_entire_pdf_sync(self, pdf_path: str) -> str:
        """
        Synchronously performs OCR on an entire PDF by converting each page to an image
        and processing them in parallel. Used as a fallback.
        """
        self._initialize_reader()

        extracted_text = []
        try:
            doc = pymupdf.open(pdf_path)
            # pages = [doc.load_page(page_num) for page_num in range(len(doc))]
            pages = []
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    if page is not None:
                        pages.append(page)
                    else:
                        logger.warning(f"Page {page_num} is None, skipping.")
                except Exception as e:
                    logger.warning(f"Failed to load page {page_num}: {e}")
            doc.close()  # Close doc after loading pages into memory

            logger.info(f"Performing full parallel image-based OCR for {len(pages)} pages.")

            # Use partial to pass the instance method to map
            ocr_results = self.executor.map(
                partial(self._ocr_pdf_page_sync, self),
                pages
            )

            for text in ocr_results:
                if text:
                    extracted_text.append(text)
        except Exception as e:
            logger.error(f"Error during _ocr_entire_pdf_sync for {pdf_path}: {e}")
            return ""

        return '\n\n'.join(extracted_text)

    # def _ocr_entire_pdf_sync(self, pdf_path: str) -> str:
    #     """
    #     Synchronously performs OCR on an entire PDF by converting each page to an image
    #     and processing them in parallel. Used as a fallback.
    #     """
    #
    #     self._initialize_reader()
    #
    #     extracted_text = []
    #     try:
    #
    #         doc = pymupdf.open(pdf_path)
    #         pages = [doc.load_page(page_num) for page_num in range(len(doc))]
    #         doc.close()
    #
    #         logger.info(f"Performing full parallel image-based OCR for {len(pages)} pages.")
    #
    #         # Use partial to pass the instance method to map
    #         ocr_results = self.executor.map(
    #             partial(self._ocr_pdf_page_sync, self),
    #             pages
    #         )
    #
    #         for text in ocr_results:
    #             if text:
    #                 extracted_text.append(text)
    #
    #     except Exception as e:
    #         logger.error(f"Error during _ocr_entire_pdf_sync for {pdf_path}: {e}")
    #         return ""
    #
    #     return '\n\n'.join(extracted_text)

    async def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image file"""
        logger.info(f"Extracting text from image: {image_path}")

        def _process_image_sync():
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Preprocess image for better OCR
            img = self._preprocess_image(img)
            return self._perform_ocr_sync(img)

        # return await asyncio.get_event_loop().run_in_executor(
        #     self.executor, _process_image
        # )
        return await asyncio.to_thread(_process_image_sync)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur for initial noise reduction. (3,3) is a good balance.
        # img = cv2.GaussianBlur(img, (1, 1), 0) # Default
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Adaptive Thresholding: More robust than simple Otsu for varying lighting
        # Block size (11) must be odd. C (2) is a constant subtracted from the mean.
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Default
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up the image (e.g., close small gaps)
        # kernel = np.ones((1, 1), np.uint8) # Default
        kernel = np.ones((2, 2), np.uint8)  # Slightly larger kernel for better effect
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Default
        img = cv2.medianBlur(img, 3)  # Additional blur to reduce salt-and-pepper noise

        # Optional: Skew correction can be added here if documents are often rotated.
        # This would require more complex image analysis (e.g., Hough lines).

        return img

    def _perform_ocr_sync(self, img: np.ndarray) -> str:
        """Synchronously performs OCR on a preprocessed image."""
        self._initialize_reader()

        try:
            # Perform OCR
            results = self.reader.readtext(img, detail=1)  # detail=1 gives bbox, text, confidence

            # Extract text with confidence filtering
            extracted_text = []
            for (bbox, text, confidence) in results:
                # Filter out low-confidence results to reduce noise
                if confidence > 0.4:  # Filter out low-confidence results # Default = 0.5
                    extracted_text.append(text)

            raw_text = ' '.join(extracted_text)
            logger.debug(f"Raw OCR text before correction: {raw_text}")

            # Apply spell correction conditionally
            if settings.OCR_ENABLE_SPELL_CORRECTION:
                cleaned_text = self.correct_thai_text(raw_text)
                logger.debug(f"Corrected text: {cleaned_text}")
                return cleaned_text
            else:
                logger.debug("Spell correction disabled.")
                return raw_text

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""

    # async def perform_ocr(self, img: np.ndarray, preprocess: bool = True) -> str:
    #     """
    #     Async OCR on an image array.
    #     Runs heavy computation in a thread pool for true async compatibility.
    #     """
    #     loop = asyncio.get_running_loop()
    #     return await loop.run_in_executor(
    #         self.executor,
    #         self._perform_ocr_sync,
    #         img,
    #         preprocess
    #     )

    async def _read_text_file(self, file_path: str) -> str:
        """Reads text from a plain text file asynchronously."""
        logger.info(f"Reading text file: {file_path}")

        def _read_file_sync():
            encodings = ['utf-8', 'utf-16', 'cp874', 'latin1']  # cp874 for Thai Windows encoding

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode {file_path} with {encoding}, trying next.")
                    continue
                except Exception as e:
                    logger.error(f"Error reading file with {encoding}: {e}")
                    continue

            raise ValueError(f"Could not decode file {file_path} with any supported encoding")

        # return await asyncio.get_event_loop().run_in_executor(
        #     self.executor, _read_file_sync
        # )
        return await asyncio.to_thread(_read_file_sync)

    async def extract_with_confidence(self, file_path: str) -> Tuple[str, List[float]]:
        """Extracts text with confidence scores for image files."""
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            def _process_with_confidence_sync():
                self._initialize_reader()
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Could not load image for confidence extraction: {file_path}")
                img = self._preprocess_image(img)

                # results = self.reader.readtext(img)
                results = self.reader.readtext(img, detail=1)

                texts = []
                confidences = []

                # for (bbox, text, confidence) in results:
                #     texts.append(text)
                #     confidences.append(confidence)
                for (bbox, text, confidence) in results:
                    texts.append(text)
                    confidences.append(float(confidence))  # Ensure confidence is float

                return ' '.join(texts), confidences

            # return await asyncio.get_event_loop().run_in_executor(
            #     self.executor, _process_with_confidence
            # )
            return await asyncio.to_thread(_process_with_confidence_sync)
        else:
            # For PDF and text files, extract text but return dummy confidence.
            # More granular confidence for these types would require parsing the text
            # and assigning a confidence score to each segment.
            text = await self.extract_text(file_path)
            logger.info(f"Extracting with confidence for non-image file. Returning text with dummy confidence.")

            print("Extracted Text:")
            print(text)
            return text, [1.0] if text else []  # Return [1.0] only if text is not empty

    def cleanup(self):
        """Clean up resources and shut down the ThreadPoolExecutor."""
        # if self.executor:
        #     self.executor.shutdown(wait=False)
        #     self.executor = None
        # self.reader = None
        if self.executor:
            logger.info("Shutting down ThreadPoolExecutor...")
            self.executor.shutdown(wait=True)  # Wait for all tasks to complete
            self.executor = None
            logger.info("ThreadPoolExecutor shut down.")
        self.reader = None
        logger.info("EasyOCR reader reset.")

    # def correct_thai_text(self, text: str) -> str:
    #     """
    #     Correct Thai text using pythainlp spell correction.
    #     This function uses the pythainlp library to correct common spelling mistakes in Thai text.
    #
    #     Args:
    #         text (str): The input Thai text to be corrected.
    #     Returns:
    #         str: The corrected Thai text.
    #     -----------
    #     Example:
    #         >>> ocr_service = OCRService()
    #         >>> corrected_text = ocr_service.correct_thai_text("สวัสดีครับ")
    #         >>> print(corrected_text)
    #         "สวัสดีครับ"
    #     -----------
    #     """
    #
    #     tokens = text.split()
    #     return ' '.join([correct(token) for token in tokens])
