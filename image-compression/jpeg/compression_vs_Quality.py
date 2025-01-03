import os
import matplotlib.pyplot as plt
from PIL import Image
from inbui import jpeg_compression  # Assuming REAL JPEG functions are in `real_jpeg.py`
from execution import load_and_pad_image, process_image  # Assuming OUR JPEG functions are in `our_jpeg.py`

def plot_compression_ratio_vs_quality(image_files, Q_values, output_dir):
    """
    Plot Compression Ratio vs Quality Factor for both REAL JPEG and OUR JPEG implementations.

    Args:
        image_files (list): List of image filenames to process.
        Q_values (list): List of quality values for compression.
        output_dir (str): Directory to save comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_file in image_files:
        print(f"Processing Compression Ratio vs Quality for {image_file}...")

        # Load image for compression
        padded_matrix, original_image = load_and_pad_image(image_file)
        if padded_matrix is None or original_image is None:
            print(f"Skipping {image_file} due to load failure.")
            continue

        # Get image dimensions (width x height)
        image = Image.open(image_file)
        m, n = image.width, image.height
        total_pixels = m * n

        # Initialize data storage for compression ratios
        real_compression_ratios = []
        our_compression_ratios = []

        for Q in Q_values:
            print(f"Processing Q={Q}...")

            # REAL JPEG processing
            real_output_file = f"real_compressed_Q{Q}.jpg"
            real_compressed_size = jpeg_compression(image_file, real_output_file, Q)  # In bytes

            # Calculate Compression Ratio for REAL JPEG
            if real_compressed_size > 0:  # Avoid division by zero
                real_compression_ratio = total_pixels / real_compressed_size
            else:
                real_compression_ratio = 0  # Invalid case
            real_compression_ratios.append(real_compression_ratio)

            # OUR JPEG processing
            our_output_dir = os.path.join(output_dir, "our_results")
            os.makedirs(our_output_dir, exist_ok=True)

            our_results = process_image(image_file, padded_matrix, original_image, [Q], our_output_dir)
            our_compressed_size = our_results["bpp"][0] * total_pixels / 8  # Convert bits to bytes

            if our_compressed_size > 0:  # Avoid division by zero
                our_compression_ratio = total_pixels / our_compressed_size
            else:
                our_compression_ratio = 0  # Invalid case
            our_compression_ratios.append(our_compression_ratio)

        # Plot Compression Ratio vs Quality Factor for the current image
        plt.figure(figsize=(10, 6))
        plt.plot(Q_values, real_compression_ratios, label="REAL JPEG", marker='o', color='blue', linestyle='-', markersize=6)
        plt.plot(Q_values, our_compression_ratios, label="OUR JPEG", marker='o', color='red', linestyle='--', markersize=6)
        plt.title(f"Compression Ratio vs Quality Factor for {image_file}")
        plt.xlabel("Quality Factor (Q)")
        plt.ylabel("Compression Ratio")
        plt.legend()
        plt.grid(True)

        plot_file = os.path.join(output_dir, f"compression_ratio_vs_quality_{image_file.split('.')[0]}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot: {plot_file}")


if __name__ == "__main__":
    # Files to process (first 4 images only)
    image_files = ["image-1.png", "image-2.png", "image-3.jpg", "image-4.jpg"]

    # JPEG quality values to test
    Q_values = [1, 3, 5, 10, 15, 25, 40, 50, 75, 90, 100]

    # Output directory for plots
    output_dir = "compression_ratio_vs_quality_results"

    # Generate Compression Ratio vs Quality Factor plots
    plot_compression_ratio_vs_quality(image_files, Q_values, output_dir)
    print("Compression Ratio vs Quality Factor plots complete.")
