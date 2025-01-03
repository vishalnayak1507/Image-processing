import os
import matplotlib.pyplot as plt
from inbui import jpeg_compression, calculate_rmse  # Assuming REAL JPEG functions are in `real_jpeg.py`
from execution import load_and_pad_image, process_image, get_image_files  # Assuming OUR JPEG functions are in `our_jpeg.py`

def compare_real_and_our_jpeg(image_files, Q_values, output_dir):
    """
    Compare REAL JPEG and OUR JPEG for the specified images.

    Args:
        image_files (list): List of image filenames to process.
        Q_values (list): List of quality values for compression.
        output_dir (str): Directory to save comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_file in image_files:
        print(f"Comparing for {image_file}...")

        # Load and pad image for OUR JPEG
        padded_matrix, original_image = load_and_pad_image(image_file)
        if padded_matrix is None or original_image is None:
            print(f"Skipping {image_file} due to load failure.")
            continue

        # Initialize data storage for both implementations
        real_jpeg_rmse = []
        real_jpeg_bpp = []
        our_jpeg_rmse = []
        our_jpeg_bpp = []

        for Q in Q_values:
            print(f"Processing Q={Q}...")

            # REAL JPEG processing
            real_output_file = f"real_compressed_Q{Q}.jpg"
            real_compressed_size = jpeg_compression(image_file, real_output_file, Q)
            real_rmse = calculate_rmse(original_image, real_output_file)
            real_jpeg_rmse.append(real_rmse)

            # Calculate BPP for REAL JPEG
            image_pixels = original_image.width * original_image.height
            real_bpp = (real_compressed_size * 8) / image_pixels
            real_jpeg_bpp.append(real_bpp)

            # OUR JPEG processing
            our_output_dir = os.path.join(output_dir, "our_results")
            os.makedirs(our_output_dir, exist_ok=True)

            our_results = process_image(image_file, padded_matrix, original_image, [Q], our_output_dir)
            our_jpeg_rmse.append(our_results["rmse"][0])  # Since Q is a single value in the loop
            our_jpeg_bpp.append(our_results["bpp"][0])

        # Plot comparison for the current image
        plt.figure(figsize=(10, 6))
        plt.plot(real_jpeg_bpp, real_jpeg_rmse, label="REAL JPEG", marker='o', color='blue', linestyle='-', markersize=6)
        plt.plot(our_jpeg_bpp, our_jpeg_rmse, label="OUR JPEG", marker='o', color='red', linestyle='--', markersize=6)
        plt.title(f"Comparison of REAL and OUR JPEG for {image_file}")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("Root Mean Square Error (RMSE)")
        plt.legend()
        plt.grid(True)

        plot_file = os.path.join(output_dir, f"comparison_{image_file.split('.')[0]}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved comparison plot: {plot_file}")


if __name__ == "__main__":
    # Files to process (first 4 images only)
    image_files = ["image-1.png", "image-2.png", "image-3.jpg", "image-4.jpg"]

    # JPEG quality values to test
    Q_values = [1, 3, 5, 10, 15, 25, 40, 50, 75, 90, 100]

    # Output directory for plots
    output_dir = "comparison_results"

    # Compare REAL JPEG and OUR JPEG implementations
    compare_real_and_our_jpeg(image_files, Q_values, output_dir)
    print("Comparison complete.")