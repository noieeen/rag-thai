from typing import List, Dict, Any
import re

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}

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

    def process_text(self, text: str) -> List[str]:
        """
        Clean and split text into sentences.
        """
        cleaned = self.clean_text(text)
        return self.split_sentences(cleaned)

    async def process_query(self, text: str) -> List[str]:
        """
        Clean and split query text into sentences (or tokens).
        """
        cleaned = self.clean_text(text)
        return self.split_sentences(cleaned)

    async def create_chunks(self, sentences: List[str], chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        Create overlapping chunks from a list of sentences.
        """
        chunks = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if current_length + len(sentence) <= chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence)
                i += 1
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(text=chunk_text, metadata={"start": i - len(current_chunk), "end": i}))
                # Overlap
                overlap_count = max(1, int(overlap / (chunk_size / len(current_chunk))))
                current_chunk = current_chunk[-overlap_count:]
                current_length = sum(len(s) for s in current_chunk)
        # Add last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(text=chunk_text, metadata={"start": len(sentences) - len(current_chunk), "end": len(sentences)}))
        return chunks