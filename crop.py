from PIL import Image

# Crop Figure 1 (Benchmarks) from the top of page 2
img2 = Image.open('pdf_pages/page_2.png')
w, h = img2.size
# Assuming standard 8.5x11 aspect ratio at 150 DPI (approx 1275 x 1650)
# Top half crop
fig1 = img2.crop((0, int(h * 0.05), w, int(h * 0.45)))
fig1.save('assets/main_benchmarks.png')

# Crop Figure 2 (Architecture Pattern) from the top of page 3
img3 = Image.open('pdf_pages/page_3.png')
w, h = img3.size
fig2 = img3.crop((0, int(h * 0.1), w, int(h * 0.4)))
fig2.save('assets/architecture_pattern.png')

print("Cropped figures successfully.")
