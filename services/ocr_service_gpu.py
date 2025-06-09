import logging
from typing import Optional

import cv2
import easyocr
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct as pythainlp_correct

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OCRServiceGPU:
    def __init__(self, use_gpu: bool = True):
        logger.info("Initializing EasyOCR with GPU=%s", use_gpu)
        self.reader = easyocr.Reader(['th', 'en'], gpu=use_gpu)

        self.use_bert = torch.cuda.is_available() and use_gpu
        if self.use_bert:
            logger.info("Loading WangchanBERTa model for spell correction...")
            self.tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
            self.model = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased").cuda()
            self.model.eval()
        else:
            logger.warning("CUDA not available or disabled. Using pythainlp for spell correction.")
            self.tokenizer = None
            self.model = None

    def extract_text(self, image_path: str) -> Optional[str]:
        logger.info("Extracting text from image: %s", image_path)
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image at %s", image_path)
            return None

        results = self.reader.readtext(image, detail=0)
        raw_text = ' '.join(results)
        logger.debug("Raw OCR result: %s", raw_text)
        return self.correct_text(raw_text)

    def correct_text(self, text: str) -> str:
        if not text.strip():
            return text

        if self.use_bert:
            logger.debug("Correcting text using WangchanBERTa...")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_ids = logits.argmax(dim=-1)
                corrected = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                return corrected
        else:
            logger.debug("Correcting text using PyThaiNLP...")
            tokens = word_tokenize(text, engine="newmm")
            corrected = ' '.join(pythainlp_correct(token) for token in tokens)
            return corrected
