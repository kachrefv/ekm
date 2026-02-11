"""
Utility for loading and extracting content from various document types.
"""
import os
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from PIL import Image
import io

class DocumentLoader:
    """Handles loading and text extraction from different file formats."""
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    async def load(self, file_path: str) -> str:
        """Load content from a file path based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.txt', '.md', '.py', '.js', '.json']:
            return self._load_text(file_path)
        elif ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
            return await self._load_image(file_path)
        else:
            # Fallback to text load for unknown extensions
            try:
                return self._load_text(file_path)
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

    def _load_text(self, file_path: str) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    async def _load_image(self, file_path: str) -> str:
        """Extract information from image using Gemini Vision."""
        if not self.llm_provider:
            raise ValueError("LLM provider required for image processing")
        
        with open(file_path, "rb") as f:
            image_data = f.read()
            
        # Use a specific prompt for image analysis
        prompt = (
            "Analyze this image and extract all relevant information, facts, and context. "
            "Describe what is happening, any text visible, and the key entities. "
            "Format the output as clear, descriptive text that can be used for knowledge extraction."
        )
        
        # We'll need to update GeminiProvider to handle this
        return await self.llm_provider.generate_from_image(image_data, prompt)

    async def load_bytes(self, content: bytes, filename: str) -> str:
        """Load content from bytes and filename."""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.txt', '.md', '.py', '.js', '.json']:
            return content.decode('utf-8', errors='ignore')
        elif ext == '.pdf':
            return self._load_pdf_bytes(content)
        elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
            return await self._load_image_bytes(content)
        else:
            try:
                return content.decode('utf-8', errors='ignore')
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

    def _load_pdf_bytes(self, content: bytes) -> str:
        """Extract text from PDF bytes."""
        stream = io.BytesIO(content)
        reader = PdfReader(stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    async def _load_image_bytes(self, content: bytes) -> str:
        """Extract information from image bytes using Gemini Vision."""
        if not self.llm_provider:
            raise ValueError("LLM provider required for image processing")
            
        prompt = (
            "Analyze this image and extract all relevant information, facts, and context. "
            "Describe what is happening, any text visible, and the key entities. "
            "Format the output as clear, descriptive text that can be used for knowledge extraction."
        )
        
        return await self.llm_provider.generate_from_image(content, prompt)
