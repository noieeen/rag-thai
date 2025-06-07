import re
from typing import List

class TextProcessor:
    """
    Service for processing and cleaning text data.
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean input text by removing unwanted characters, extra spaces, and normalizing whitespace.
        """
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7Eก-๙เ-์]', '', text)
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation.
        """
        # Basic sentence splitting for Thai and English
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[ะ-์])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words (very basic, for demonstration).
        """
        # Split by whitespace for now
        return text.split()