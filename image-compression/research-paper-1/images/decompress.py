import os
import subprocess
import shutil

# Remove and recreate the 'compressed' directory
compressed_dir = "compressed"

for i in range(4, 6):
    archive_file = f"./{compressed_dir}/im{i}.archive"
    jbg_file = f"./{compressed_dir}/m_im{i}.jbg"
    pbm_file = f"./m_im{i}.pbm"  # Save output in the compressed directory

    # Extract archive with ZPAQ
    subprocess.run(["zpaq", "x", archive_file], check=True)

    # Convert JBG back to PBM
    with open(pbm_file, "w") as pbm_out:
        subprocess.run(["jbgtopbm", jbg_file], stdout=pbm_out, check=True)