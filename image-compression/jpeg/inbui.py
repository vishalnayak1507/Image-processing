import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def jpeg_compression(input_image_path, output_image_path, quality):
    """
    Compresses an image using JPEG compression, saves the compressed file,
    and returns the compressed file size.

    Parameters:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the compressed JPEG image.
        quality (int): JPEG quality factor (1-100, higher means better quality).

    Returns:
        compressed_file_size (int): Size of the compressed file in bytes.
    """
    original_image = Image.open(input_image_path)

    # Ensure the image is in a JPEG-compatible mode
    if original_image.mode not in ("L", "RGB"):
        if original_image.mode == "RGBA" or original_image.mode.startswith("P"):
            original_image = original_image.convert("RGB")
        else:
            original_image = original_image.convert("L")

    original_image.save(output_image_path, format="JPEG", quality=quality)
    compressed_file_size = os.path.getsize(output_image_path)
    return compressed_file_size


def calculate_rmse(original_image, compressed_image_path):
    """
    Calculates RMSE between the original image and the decompressed JPEG image.

    Parameters:
        original_image (PIL.Image): The original image.
        compressed_image_path (str): Path to the compressed JPEG image.

    Returns:
        rmse (float): Root Mean Square Error between the images.
    """
    compressed_image = Image.open(compressed_image_path)
    original_array = np.array(original_image)
    compressed_array = np.array(compressed_image.resize(original_image.size))
    mse = np.mean((original_array - compressed_array) ** 2)
    return np.sqrt(mse)


def process_images(image_count):
    """
    Processes images named image-1.png to image-<image_count>.png.
    Compresses them for multiple Q values and plots RMSE vs. BPP.

    Parameters:
        image_count (int): Number of input images (e.g., 20 for image-1.png to image-20.png).
    """
    Q_values = [1, 3, 5, 10, 15, 25, 40, 50, 75, 90, 100]
    output_dir = "compressed_results"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, image_count + 1):
        input_file = f"image-{i}.png"
        print(f"Processing {input_file}...")

        try:
            original_image = Image.open(input_file).convert('L')  # Convert to grayscale if needed
        except FileNotFoundError:
            print(f"File {input_file} not found. Skipping...")
            continue

        rmse_values = []
        bpp_values = []

        for Q in Q_values:
            output_file = os.path.join(output_dir, f"image-{i}_Q{Q}.jpg")
            compressed_size = jpeg_compression(input_file, output_file, Q)
            rmse = calculate_rmse(original_image, output_file)
            rmse_values.append(rmse)

            # Calculate BPP (Bits Per Pixel)
            image_pixels = original_image.width * original_image.height
            bpp = (compressed_size * 8) / image_pixels  # Bits per pixel
            bpp_values.append(bpp)

            print(f"Image-{i}, Q={Q}, RMSE={rmse:.4f}, BPP={bpp:.4f}")

        # Plot RMSE vs BPP for the current image
        plt.figure(figsize=(10, 6))
        plt.plot(bpp_values, rmse_values, marker='o', label=f"image-{i}")
        plt.title(f"RMSE vs. BPP for image-{i}")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("Root Mean Square Error (RMSE)")
        plt.grid(True)
        plt.legend()
        plot_file = os.path.join(output_dir, f"image-{i}_rmse_vs_bpp.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot: {plot_file}")


if __name__ == "__main__":
    process_images(5)  # Adjust the range as needed (e.g., 20 for image-1.png to image-20.png)
