from PIL import Image
import os

images = [
    "/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/kv_cache_scientific_1773336832557.png",
    "/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/architecture_block_scientific_1773336847584.png",
    "/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/nvfp4_quantization_scientific_1773336860703.png"
]

for img_path in images:
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            print(f"Original size of {os.path.basename(img_path)}: {img.size}")
            # The images are likely 1024x1024.
            # To "unsquish" vertical stretching, we stretch horizontally to 1820x1024 (approx 16:9)
            new_img = img.resize((1820, 1024), Image.Resampling.LANCZOS)
            new_img.save(img_path)
            print(f"Resized {os.path.basename(img_path)} to 1820x1024")
    else:
        print(f"Skipping {img_path}, not found.")
