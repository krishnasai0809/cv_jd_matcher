import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using OCR for scanned documents.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""

    try:
        # Attempt to extract text using PyMuPDF (works for digital PDFs)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                extracted_text = page.get_text("text")
                if extracted_text.strip():  # If there's text, append it
                    text += extracted_text + "\n"

        # If no text was found, try OCR
        if not text.strip():
            print("No text found in PDF. Running OCR...")
            images = convert_from_path(pdf_path)  # Convert PDF pages to images
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

        return text.strip()

    except Exception as e:
        return f"Error processing PDF: {e}"

# Example Usage
pdf_path = "cvs/CV_Akhila_Macherla.pdf"  # Change to your file path
extracted_text = extract_text_from_pdf(pdf_path)

print("Extracted Text:\n", extracted_text)
