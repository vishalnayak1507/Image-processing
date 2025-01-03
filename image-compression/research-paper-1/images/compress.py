import os
import subprocess
import shutil

# Remove and recreate the 'compressed' directory
compressed_dir = "compressed"
if os.path.exists(compressed_dir):
    shutil.rmtree(compressed_dir)
os.makedirs(compressed_dir)

# Encode and compress images
for i in range(4, 6):
    pbm_file = f"m_im{i}.pbm"
    jbg_file = f"./{compressed_dir}/m_im{i}.jbg"
    archive_file = f"./{compressed_dir}/im{i}.archive"
    png_file = f"res_im{i}.png"

    # Convert PBM to JBG
    subprocess.run(["pbmtojbg", pbm_file, jbg_file], check=True)

    # Create archive with ZPAQ
    subprocess.run(["zpaq", "a", archive_file, jbg_file, png_file], check=True)

    # Delete intermediate files (PNG and PBM)
    if os.path.exists(pbm_file):
        os.remove(pbm_file)
    if os.path.exists(png_file):
        os.remove(png_file)

# Decompress and reconstruct images
# for i in range(4, 6):
#     archive_file = f"./{compressed_dir}/im{i}.archive"
#     jbg_file = f"./{compressed_dir}/m_im{i}.jbg"
#     pbm_file = f"./{compressed_dir}/m_im{i}.pbm"  # Save output in the compressed directory
#     png_file = f"./{compressed_dir}/res_im{i}.png"  # Expected PNG file path

#     # Extract archive with ZPAQ
#     subprocess.run(["zpaq", "x", archive_file], check=True)

#     # Convert JBG back to PBM
#     with open(pbm_file, "w") as pbm_out:
#         subprocess.run(["jbgtopbm", jbg_file], stdout=pbm_out, check=True)

