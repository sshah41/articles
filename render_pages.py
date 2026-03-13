import fitz
import os

pdf_path = "NVIDIA-Nemotron-3-Super-Technical-Report.pdf"
out_dir = "pdf_pages"
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
gallery_html = "<html><body style='font-family:sans-serif; background:#eee; text-align:center;'>"
for i in range(len(doc)):
    page = doc.load_page(i)
    pix = page.get_pixmap(dpi=150)
    img_name = f"page_{i+1}.png"
    pix.save(os.path.join(out_dir, img_name))
    gallery_html += f"<div><h2>Page {i+1}</h2><img src='pdf_pages/{img_name}' style='border:1px solid #ccc; max-width:800px; margin-bottom: 20px;'></div>"

gallery_html += "</body></html>"
with open("page_gallery.html", "w") as f:
    f.write(gallery_html)
print(f"Rendered {len(doc)} pages to {out_dir} and created page_gallery.html")
