import fitz
import os

pdf_path = "NVIDIA-Nemotron-3-Super-Technical-Report.pdf"
out_dir = "pdf_extract"

doc = fitz.open(pdf_path)
count = 0
for i in range(len(doc)):
    for img in doc.get_page_images(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(os.path.join(out_dir, f"img_{i}_{xref}.png"))
        count += 1
print(f"Extracted {count} images.")
