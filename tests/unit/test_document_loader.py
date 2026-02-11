import pytest
import os
import io
from unittest.mock import MagicMock, patch, AsyncMock
from ekm.utils.document_loader import DocumentLoader

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_from_image = AsyncMock(return_value="Extracted text from image")
    return llm

@pytest.mark.asyncio
async def test_load_text(tmp_path):
    d = tmp_path / "test.txt"
    d.write_text("Hello World", encoding="utf-8")
    
    loader = DocumentLoader()
    content = await loader.load(str(d))
    assert content == "Hello World"

@pytest.mark.asyncio
async def test_load_pdf(tmp_path):
    # Mocking PdfReader to avoid needing a real PDF file
    with patch("ekm.utils.document_loader.PdfReader") as MockReader:
        mock_reader = MockReader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Content"
        mock_reader.pages = [mock_page]
        
        loader = DocumentLoader()
        # We just need a path that ends in .pdf
        content = await loader.load("dummy.pdf")
        assert content == "PDF Content\n"

@pytest.mark.asyncio
async def test_load_image(mock_llm):
    with patch("builtins.open", MagicMock(return_value=io.BytesIO(b"fake_image_data"))):
        loader = DocumentLoader(llm_provider=mock_llm)
        content = await loader.load("test.jpg")
        assert content == "Extracted text from image"
        mock_llm.generate_from_image.assert_called_once()

@pytest.mark.asyncio
async def test_load_bytes_text():
    loader = DocumentLoader()
    content = await loader.load_bytes(b"Byte Content", "test.txt")
    assert content == "Byte Content"

@pytest.mark.asyncio
async def test_load_bytes_pdf():
    with patch("ekm.utils.document_loader.PdfReader") as MockReader:
        mock_reader = MockReader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Byte Content"
        mock_reader.pages = [mock_page]
        
        loader = DocumentLoader()
        content = await loader.load_bytes(b"%PDF-1.4...", "test.pdf")
        assert content == "PDF Byte Content\n"
