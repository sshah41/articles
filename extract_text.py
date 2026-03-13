import fitz
import sys

pdf_path = "NVIDIA-Nemotron-3-Super-Technical-Report.pdf"
doc = fitz.open(pdf_path)
with open("pdf_text.txt", "w", encoding="utf-8") as f:
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        f.write(f"--- PAGE {i+1} ---\n{text}\n")
print(f"Extracted text from {len(doc)} pages.")
