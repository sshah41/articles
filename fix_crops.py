import fitz

doc = fitz.open("NVIDIA-Nemotron-3-Super-Technical-Report.pdf")

# Fix Figure 2 (Architecture Pattern) on Page 3 (index 2)
page3 = doc.load_page(2)
blocks = page3.get_text("dict")["blocks"]

top_y = 0
bottom_y = page3.rect.height

for b in blocks:
    if "lines" not in b: continue
    text = "".join([s["text"] for l in b["lines"] for s in l["spans"]])
    if "2.1. Model Architecture" in text:
        top_y = b["bbox"][3]
    if "Figure 2 | Nemotron 3 Super layer pattern" in text:
        bottom_y = b["bbox"][1]

print(f"Arch Pattern BBox: top={top_y}, bottom={bottom_y}")
# Add a small padding
rect = fitz.Rect(0, top_y - 5, page3.rect.width, bottom_y + 5)
pix = page3.get_pixmap(dpi=300, clip=rect)
pix.save("assets/architecture_pattern.png")

# Fix Figure 1 (Benchmarks) on Page 2 (index 1)
page2 = doc.load_page(1)
blocks = page2.get_text("dict")["blocks"]
top_y = 0
bottom_y = page2.rect.height
for b in blocks:
    if "lines" not in b: continue
    text = "".join([s["text"] for l in b["lines"] for s in l["spans"]])
    if "Nemotron 3 Super : Open, Efficient" in text:
        top_y = b["bbox"][3]
    if "Figure 1 | Accuracy and throughput" in text:
        bottom_y = b["bbox"][1]

print(f"Bench BBox: top={top_y}, bottom={bottom_y}")
rect = fitz.Rect(0, top_y - 5, page2.rect.width, bottom_y + 5)
pix = page2.get_pixmap(dpi=300, clip=rect)
pix.save("assets/main_benchmarks.png")

# For MTP and NVFP4, let's just use the raw images we already had, they looked fine.
# But let's check latent moe architecture. The user said it was cut off.
# latent moe is img_4_98.png. The user's screenshot showed text overlapping. 
# It's an issue with markdown formatting, not the image itself, but we will fix the markdown.
