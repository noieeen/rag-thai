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

from resource import getrusage, RUSAGE_SELF
import gc
from utils.memory_utils import MemoryGuard

from core.config import settings

# pythainlp is used for spell correction
from pythainlp.spell import correct
from pythainlp.tokenize import word_tokenize, sent_tokenize

from attacut import tokenize as attacut_tokenize

from functools import lru_cache, partial
import pytesseract

import re
from pythainlp.corpus.common import thai_words

DICT_WORD = set(thai_words())
## AI
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
# model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").cuda()
# CPU
# model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").to("cpu")


logger = logging.getLogger(__name__)

CUSTOM_FIX = {
    "ตู": "ดู",
    "ใกล้ซิด": "ใกล้ชิด",
    "ข้า": "ค้า",
    "ยอต": "ยอด",
    "ทาร": "การ",
    "ทํา": "ทำ",
    "กอยุทธ์": "กลยุทธ์"
}

# Use regex or known UI words like “Posted:”, “Let’s Chat”, etc.
# No need word
LAYOUT_NOISE = ["Let's Chat", "Posted:", "บทความอื่นๆ ที่เกี่ยวข้อง", "สนใจคลิกเลย"]


class OCRService:
    """
    Service for OCR text extraction from images and PDFs,
    optimized for speed and accuracy.
    """

    def __init__(self, max_workers=None,
                 max_memory_gb=settings.OCR_PROCESSING_MAX_MEMORY_GB):  ## os.cpu_count() -> all cpu / too much consume memory

        pytesseract.pytesseract.tesseract_cmd = f'{settings.OCR_TESSERACT_EXECUTE_LOCATION}'

        # Calculate workers based on available memory
        if max_workers is None:
            max_workers = max(1, int(max_memory_gb * 0.8))  # 80% of available memory

        self.max_memory_gb = max_memory_gb
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = 0
        self.memory_semaphore = asyncio.Semaphore(max_workers * 2)  # Control concurrent heavy operations

        self.reader = None
        logger.info(f"Initialized OCRService with {max_workers} thread pool workers.")

    async def _memory_safe_ocr(self, img: np.ndarray) -> str:
        """Process OCR with memory constraints"""
        async with self.memory_semaphore:
            if self._check_memory_usage() > self.max_memory_gb * 0.9:  # 90% threshold
                gc.collect()
                await asyncio.sleep(1)  # Allow memory to settle
            return await asyncio.to_thread(self._perform_ocr_sync, img)

    def _check_memory_usage(self) -> float:
        """Check current memory usage in GB"""
        return getrusage(RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # Convert to GB

    def _initialize_reader(self):
        """Lazy-load and initialize EasyOCR reader."""
        if self.reader is None:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(
                settings.OCR_LANGUAGES,
                gpu=settings.OCR_GPU
            )
            logger.info("EasyOCR reader initialized successfully.")

    @lru_cache(maxsize=4096)  # Cache up to 2048 unique word corrections
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
        # tokens = text.split()

        # Apply cached correction to each token
        tokens = word_tokenize(text, engine=settings.THAI_TOKENIZER)
        corrected_tokens = [self._cached_correct_word(token) for token in tokens]

        return ' '.join(corrected_tokens)

    # def _chunk_and_correct(self, raw_text: str, spell_correct_fn) -> str:
    #     sentences = sent_tokenize(raw_text, engine="whitespace+newline")
    #     filtered = [s for s in sentences if not any(n in s for n in LAYOUT_NOISE)]
    #
    #     cleaned = []
    #     for sentence in filtered:
    #         tokens = attacut_tokenize(sentence)
    #         corrected = [spell_correct_fn(t) for t in tokens]
    #         cleaned.append(' '.join(corrected))
    #
    #     return '\n'.join(cleaned)
    def _chunk_clean_spell(self, text: str, spell_fn) -> str:

        # 1. Normalize and clean layout noise
        text = re.sub(r'\n+', '\n', text)
        for noise in LAYOUT_NOISE:
            text = text.replace(noise, '')

        # 2. Sentence segmentation
        sentences = sent_tokenize(text, engine="whitespace+newline")
        cleaned = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            tokens = attacut_tokenize(sentence)

            # 3. Only correct tokens not in dictionary
            corrected = [
                spell_fn(t) if t not in DICT_WORD and len(t.strip()) > 1 else t
                for t in tokens
            ]
            cleaned.append(' '.join(corrected))

        return '\n'.join(cleaned)

    def _correct_token_gpu(self, token: str, top_k: int = 1) -> str:
        if token in DICT_WORD or len(token.strip()) < 2:
            return token

        # if torch.cuda.is_available():
        #     logger.info("CUDA available. Loading model on GPU.")
        #     model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").cuda()
        # else:
        #     import platform
        #     system = platform.system()
        #     if system == "Darwin":
        #         logger.warning("macOS detected: CUDA is not supported. Using CPU.")
        #     else:
        #         logger.warning("CUDA not available. Falling back to CPU.")
        #     model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").to("cpu")

        ## Check Device
        try:
            if torch.cuda.is_available():
                model_device = torch.device("cuda")
                logger.info("CUDA is available. Loading model on GPU.")
            else:
                import platform
                if platform.system() == "Darwin":
                    logger.warning("macOS detected: CUDA is not supported. Using CPU.")
                else:
                    logger.warning("CUDA not available. Using CPU.")
                model_device = torch.device("cpu")
            model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").to(
                model_device)
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            model_device = torch.device("cpu")
            model = None

        try:
            # # input_ids = tokenizer.encode(f"[MASK]", return_tensors="pt").cuda()
            # input_ids = tokenizer.encode(f"[MASK]", return_tensors="pt").to(model_device)
            #
            # # Guard clause: ensure the mask token index is not empty
            # mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            # if mask_token_index.numel() == 0:
            #     raise ValueError("No [MASK] token found in input_ids. Skipping token.")
            # Explicitly include the [MASK] token in the input
            input_ids = tokenizer.encode(f"{token} [MASK]", return_tensors="pt").to(model_device)
            # Validation: Check for [MASK] token presence
            if tokenizer.mask_token_id not in input_ids:
                raise ValueError("No [MASK] token found in input_ids. Skipping token.")

            with torch.no_grad():
                logits = model(input_ids).logits
                mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                probs = logits[0, mask_token_index].softmax(dim=-1)
                top_tokens = torch.topk(probs, k=top_k).indices[0].tolist()
                return tokenizer.decode([top_tokens[0]]).strip()
        except Exception as e:
            # logger.warning(f"GPU spell correction failed for '{token}': {e}")
            logger.warning(f"GPU spell correction failed for '{token}': {type(e).__name__}: {e}")
            return token

    def _correct_with_transformer(self, text: str, max_tokens=15) -> str:
        chunks = attacut_tokenize(text)
        cleaned_chunks = []

        for i in range(0, len(chunks), max_tokens):
            chunk = " ".join(chunks[i:i + max_tokens])
            corrected = self._correct_token_gpu(chunk)
            cleaned_chunks.append(corrected)

        return " ".join(cleaned_chunks)

    def _clean_and_correct_text(self, raw_text: str) -> str:
        # 1. ล้าง whitespace + newline
        text = raw_text.replace("\n", " ").replace("  ", " ").strip()

        # 2. ตัดคำด้วย attacut
        tokens = attacut_tokenize(text)

        # 3. กรอง layout noise
        tokens = [t for t in tokens if all(n not in t for n in LAYOUT_NOISE)]

        # 4. แก้คำผิดแบบ caching
        if settings.OCR_ENABLE_SPELL_CORRECTION:
            corrected = [self._cached_correct_word(t) for t in tokens]
        else:
            corrected = tokens

        return ' '.join(corrected)

    def _apply_custom_fixes(self, text: str) -> str:
        for k, v in CUSTOM_FIX.items():
            text = text.replace(k, v)
        return text

    def correct_tokens_fast(self, tokens: List[str], processes: int = 4) -> List[str]:
        """
        Fast spell correction using multiprocessing + LRU cache.
        :param tokens: list of tokens to correct
        :param processes: number of parallel processes
        """
        from multiprocessing import Pool, get_context

        if not tokens:
            return []

        try:
            with get_context("fork").Pool(processes=processes) as pool:
                corrected = pool.map(self._cached_correct_word, tokens)
            return corrected
        except Exception as e:
            logger.warning(f"Multiprocessing spell correction failed: {e}. Falling back to single-threaded.")
            return [self._cached_correct_word(t) for t in tokens]

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a file (PDF, image, or plain text)."""
        logger.info(f"Starting text extraction for: {file_path}")
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path_obj.suffix.lower()

        if file_extension == '.pdf':
            return await self._extract_from_pdf(str(file_path_obj))
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return await self._extract_from_image(str(file_path_obj))
        elif file_extension == '.txt':
            return await self._read_text_file(str(file_path_obj))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    # async def _extract_from_pdf(self, pdf_path: str) -> str:
    #     """
    #     Extract text from a PDF file. Prioritizes direct text extraction.
    #     Falls back to parallel OCR for image-based pages.
    #     """
    #     logger.info(f"Extracting text from PDF: {pdf_path}")
    #
    #     def _process_pdf_sync():
    #         extracted_text_parts = []
    #         pages_to_ocr_images = []
    #
    #         try:
    #             doc = pymupdf.open(pdf_path)
    #             for page_num in range(len(doc)):
    #                 try:
    #                     page = doc.load_page(page_num)
    #                     if page is None: # Explicit check for None page
    #                         logger.warning(f"Page {page_num + 1} of {pdf_path} is None. Skipping for direct text/OCR.")
    #                         continue
    #
    #                     text = page.get_text()
    #                     if text.strip():
    #                         extracted_text_parts.append(text)
    #                     else:
    #                         logger.info(f"No direct text on page {page_num + 1}. Preparing for OCR.")
    #                         pages_to_ocr_images.append(page)
    #                 except Exception as page_e:
    #                     logger.error(f"Error loading or processing page {page_num + 1}: {page_e}. Skipping this page.")
    #                     continue # Continue to the next page even if one fails
    #             doc.close()
    #
    #             # Perform OCR on pages that didn't yield text, in parallel
    #             if pages_to_ocr_images:
    #                 logger.info(f"Performing parallel OCR on {len(pages_to_ocr_images)} image-based PDF pages.")
    #                 # Use partial to pass the instance method along with the page object to map
    #                 ocr_results = list(self.executor.map(
    #                     partial(self._ocr_pdf_page_sync, self),
    #                     pages_to_ocr_images
    #                 ))
    #
    #                 for ocr_text in ocr_results:
    #                     if ocr_text:
    #                         extracted_text_parts.append(ocr_text)
    #
    #             logger.info(f"Finished processing PDF: {pdf_path}")
    #             return '\n\n'.join(extracted_text_parts)
    #
    #         except Exception as e:
    #             logger.error(f"Critical error opening/processing PDF {pdf_path}: {e}. Attempting full image-based OCR fallback.")
    #             return self._ocr_entire_pdf_sync(pdf_path)
    #
    #     return await asyncio.to_thread(_process_pdf_sync)

    async def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with automatic fallback to serial processing"""
        logger.info(f"Extracting text from PDF: {pdf_path}")

        def _process_pdf_sync():
            try:
                # First attempt with parallel processing
                result = self._try_parallel_pdf_processing(pdf_path)
                if result:  # If we got any text, return it
                    return result

                # Fallback to serial processing
                logger.warning("Parallel processing failed, attempting serial processing")
                return self._process_pdf_serially(pdf_path)

            except Exception as e:
                logger.error(f"PDF processing completely failed: {e}")
                return ""

        return await asyncio.to_thread(_process_pdf_sync)

    def _try_parallel_pdf_processing(self, pdf_path: str) -> str:
        """Attempt parallel processing of PDF"""
        extracted_text_parts = []
        pages_to_ocr_images = []

        try:
            doc = pymupdf.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                if page is None:
                    continue

                text = page.get_text()
                if text.strip():
                    extracted_text_parts.append(text)
                else:
                    pages_to_ocr_images.append(page)

            if pages_to_ocr_images:
                # Try parallel processing
                ocr_results = list(self.executor.map(
                    partial(self._ocr_pdf_page_sync, self),
                    pages_to_ocr_images
                ))
                extracted_text_parts.extend(r for r in ocr_results if r)

            return '\n\n'.join(extracted_text_parts)
        except Exception as e:
            logger.error(f"Parallel PDF processing failed: {e}")
            return ""

    def _process_pdf_serially(self, pdf_path: str) -> str:
        """Process PDF serially as fallback"""
        extracted_text_parts = []

        try:
            doc = pymupdf.open(pdf_path)
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    if page is None:
                        logger.warning(f"Page {page_num + 1} is None in serial processing")
                        continue

                    text = page.get_text()
                    if text.strip():
                        extracted_text_parts.append(text)
                    else:
                        # Process immediately in same thread
                        ocr_text = self._ocr_pdf_page_sync(self, page)
                        if ocr_text:
                            extracted_text_parts.append(ocr_text)
                except Exception as page_e:
                    logger.error(f"Error processing page {page_num + 1} serially: {page_e}")

            return '\n\n'.join(extracted_text_parts)
        except Exception as e:
            logger.error(f"Serial PDF processing failed: {e}")
            return ""

    # @staticmethod
    # def _ocr_pdf_page_sync(instance: 'OCRService', page) -> str:
    #     """
    #     Synchronously performs OCR on a single PDF page image.
    #     Requires an instance of OCRService to access its _perform_ocr_sync method.
    #     """
    #     if page is None: # Defensive check
    #         logger.error("Attempted OCR on a None PDF page object.")
    #         return ""
    #     try:
    #         # Render page to a high-resolution pixmap for better OCR
    #         mat = pymupdf.Matrix(2, 2) # 2x zoom for better quality
    #         pix = page.get_pixmap(matrix=mat)
    #
    #         # Convert pixmap to numpy array via bytes
    #         img_data = pix.tobytes("png")
    #         img_array = np.frombuffer(img_data, dtype=np.uint8)
    #         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #
    #         if img is None:
    #             logger.error(f"Failed to decode image from PDF page: {page.number if hasattr(page, 'number') else 'unknown'}.")
    #             return ""
    #
    #         return instance._perform_ocr_sync(img)
    #     except Exception as e:
    #         logger.error(f"Error during OCR of PDF page {page.number if hasattr(page, 'number') else 'unknown'}: {e}")
    #         return ""

    @staticmethod
    def _ocr_pdf_page_sync(instance: 'OCRService', page) -> str:
        """
        Synchronously performs OCR on a single PDF page image with improved error handling.
        """
        if page is None:
            logger.error("Attempted OCR on a None PDF page object.")
            return ""

        try:
            # First validate the page exists and is accessible
            if not hasattr(page, 'get_pixmap'):
                logger.error(f"Invalid page object encountered: {page}")
                return ""

            # Try to get basic page info to validate it's accessible
            try:
                _ = page.rect  # This will fail if page is corrupted
            except Exception as e:
                logger.error(f"Page validation failed, likely corrupted: {e}")
                return ""

            # Render page to a high-resolution pixmap for better OCR
            mat = pymupdf.Matrix(2, 2)  # 2x zoom for better quality
            try:
                pix = page.get_pixmap(matrix=mat)
            except Exception as e:
                logger.error(f"Failed to render page to pixmap: {e}. Trying with lower quality...")
                # Fallback to lower quality render
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
                except Exception as fallback_e:
                    logger.error(f"Fallback render also failed: {fallback_e}")
                    return ""

            # Convert pixmap to numpy array via bytes
            try:
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    logger.error("Failed to decode image from PDF page.")
                    return ""

                return instance._perform_ocr_sync(img)
            except Exception as e:
                logger.error(f"Error during image processing: {e}")
                return ""

        except Exception as e:
            logger.error(f"Unexpected error during OCR of PDF page: {e}")
            return ""

    def _ocr_entire_pdf_sync(self, pdf_path: str) -> str:
        """
        Synchronously performs OCR on an entire PDF by converting each page to an image
        and processing them in parallel. Used as a fallback when direct text extraction fails.
        """
        self._initialize_reader()

        extracted_text = []
        try:
            doc = pymupdf.open(pdf_path)
            valid_pages = []
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    if page is not None:
                        valid_pages.append(page)
                    else:
                        logger.warning(f"Page {page_num + 1} of {pdf_path} is None, skipping during full OCR fallback.")
                except Exception as e:
                    logger.warning(f"Failed to load page {page_num + 1} for full OCR fallback: {e}. Skipping.")
            doc.close()  # Close doc after loading pages into memory

            if not valid_pages:
                logger.warning(f"No valid pages found in PDF {pdf_path} for full OCR fallback.")
                return ""

            logger.info(f"Performing full parallel image-based OCR for {len(valid_pages)} pages.")

            # Use partial to pass the instance method to map
            ocr_results = self.executor.map(
                partial(self._ocr_pdf_page_sync, self),
                valid_pages
            )

            for text in ocr_results:
                if text:
                    extracted_text.append(text)
        except Exception as e:
            logger.error(f"Error during _ocr_entire_pdf_sync for {pdf_path}: {e}")
            return ""

        return '\n\n'.join(extracted_text)

    async def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image file."""
        logger.info(f"Extracting text from image: {image_path}")

        # img = Image.open(image_path)
        # print(pytesseract.image_to_string(img, lang=settings.OCR_TESSERACT_LANGUAGE))
        cv_img = cv2.imread(image_path)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        text = pytesseract.image_to_string(thresh, config=settings.OCR_TESSERACT_CONFIG,
                                           lang=settings.OCR_TESSERACT_LANGUAGE)
        text = text.encode('utf-8', 'replace').decode()
        # cleaned_text = self._clean_and_correct_text(text)
        # cleaned_text = self._apply_custom_fixes(text)

        # cleaned_text = self._chunk_clean_spell(text, correct) # -> not bad

        cleaned_text = self._correct_with_transformer(text)
        #
        # text = text.replace("\n", " ").replace("  ", " ").strip()
        # tokens = attacut_tokenize(text)
        #
        # # Try Use regex or keyword filters to skip boilerplate text
        # LAYOUT_NOISE = ["Let's Chat", "Posted:", "@", "Festival", "Songkran", "ChocoCRM", "บทความ", "แบรนด์", "แคมเปญ"]
        # tokens = [t for t in tokens if t.strip() and all(noise not in t for noise in LAYOUT_NOISE)]
        #
        # # 4. fix wrong word and caching
        # if settings.OCR_ENABLE_SPELL_CORRECTION:
        #     corrected = [self._cached_correct_word(t) for t in tokens]
        # else:
        #     corrected = tokens
        # cleaned_text = ' '.join(corrected)
        # # corrected = [self._cached_correct_word(t) for t in tokens]
        # # ' '.join(corrected)
        print(cleaned_text)

        def _process_image_sync():
            return "XX"
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            img = self._preprocess_image(img)
            return self._perform_ocr_sync(img)

        return await asyncio.to_thread(_process_image_sync)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocesses image for better OCR accuracy."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (3, 3), 0)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.medianBlur(img, 3)

        return img

    def _perform_ocr_sync(self, img: np.ndarray) -> str:
        """Synchronously performs OCR on a preprocessed image using Tesseract only."""
        try:
            text = pytesseract.image_to_string(img, config="-l tha+eng --psm 6")
            text = text.encode('utf-8', 'replace').decode()

            cleaned_text = self._clean_and_correct_text(text)
            cleaned_text = self._apply_custom_fixes(cleaned_text)

            return cleaned_text
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def _perform_tesseract_ocr(self, img: np.ndarray) -> str:
        config = "-l tha+eng --psm 6"
        try:
            text = pytesseract.image_to_string(img, config=config)
            logger.debug(f"Tesseract OCR output: {text}")
            if settings.OCR_ENABLE_SPELL_CORRECTION:
                return self.correct_thai_text(text)
            return text
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    async def _read_text_file(self, file_path: str) -> str:
        """Reads text from a plain text file asynchronously."""
        logger.info(f"Reading text file: {file_path}")

        def _read_file_sync():
            encodings = ['utf-8', 'utf-16', 'cp874', 'latin1']

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

                results = self.reader.readtext(img, detail=1)

                texts = []
                confidences = []

                for (bbox, text, confidence) in results:
                    texts.append(text)
                    confidences.append(float(confidence))

                return ' '.join(texts), confidences

            return await asyncio.to_thread(_process_with_confidence_sync)
        else:
            text = await self.extract_text(file_path)
            logger.info(f"Extracting with confidence for non-image file. Returning text with dummy confidence.")
            return text, [1.0] if text else []

    def cleanup(self):
        """Clean up resources and shut down the ThreadPoolExecutor."""
        if self.executor:
            logger.info("Shutting down ThreadPoolExecutor...")
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("ThreadPoolExecutor shut down.")
        self.reader = None
        logger.info("EasyOCR reader reset.")
