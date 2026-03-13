import os
import shutil
import glob

files = glob.glob("imagesfrompaper/*.png")

# Mapping the filenames to cleaner asset names based on content
for f in files:
    if "9.25.04" in f: shutil.copy(f, "assets/table_1_architecture.png")
    elif "9.25.38" in f: shutil.copy(f, "assets/figure_3_latent_moe.png")
    elif "9.31.00" in f: shutil.copy(f, "assets/table_9_comprehensive_eval.png")
    elif "9.26.57" in f: shutil.copy(f, "assets/figure_4_5_mtp.png")
    elif "9.24.29" in f: shutil.copy(f, "assets/figure_1_benchmarks.png")
    elif "9.25.15" in f: shutil.copy(f, "assets/figure_2_layer_pattern.png")

print("Copied images successfully.")
