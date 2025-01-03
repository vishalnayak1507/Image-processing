import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from compression_pipeline import compress_image, decompress_image
from quantization_matrices import standard_luminance  # Import the standard matrix


def get_image_files():
    """
    Generate a list of image filenames (image-1.png to image-20.png).

    Returns:
        list: List of image filenames.
    """
    # image_files = [f"image-{i}.{'png' if i <= 2 else 'jpg'}" for i in range(1, 21)]
    image_files = [f"image-{i}.jpg" for i in range(23, 24)]
    return image_files


def load_and_pad_image(image_file):
    """
    Load an image and pad its dimensions to the nearest multiple of 8.

    Args:
        image_file (str): Path to the image file.

    Returns:
        tuple: A tuple of (padded_matrix, original_image).
    """
    try:
        image = Image.open(image_file).convert('L')  # Convert to grayscale
        output_filename = 'hello3.jpg'
        image.save(output_filename)
    except FileNotFoundError:
        print(f"File {image_file} not found. Skipping...")
        return None, None

    new_width = (image.width // 8 + 1) * 8
    new_height = (image.height // 8 + 1) * 8
    # print(f"Resized to: {new_width} x {new_height}")

    padded_matrix = np.zeros((new_height, new_width), dtype=np.int32)
    padded_matrix[:image.height, :image.width] = np.array(image)
    return padded_matrix, image


def process_image(image_file, padded_matrix, original_image, Q_values, output_dir):
    """
    Process a single image: compress, decompress, calculate RMSE and BPP.

    Args:
        image_file (str): Image filename.
        padded_matrix (np.array): Padded image matrix.
        original_image (PIL.Image): Original image.
        Q_values (list): List of quality values.
        output_dir (str): Directory to save results.

    Returns:
        dict: A dictionary with RMSE and BPP data for the image.
    """
    image_rmse = []
    image_bpp = []

    for Q in Q_values:
        factor = 50.0 / Q
        quant_matrix = np.maximum(1, np.round(standard_luminance * factor)).astype(np.int32)
        print(f"Processing with Q={Q} (factor={factor:.2f})...")

        filename = compress_image(padded_matrix, quant_matrix)
        restored_image = decompress_image(filename)
        restored_image = restored_image[:original_image.height, :original_image.width]

        mse = np.mean((np.array(original_image) - restored_image) ** 2)
        rmse = np.sqrt(mse)
        image_rmse.append(rmse)
        print(f'{rmse} and {Q}')

        compressed_file_size = os.path.getsize(filename) * 8  # Size in bits
        bpp = compressed_file_size / (original_image.height * original_image.width)
        image_bpp.append(bpp)

        restored_image_pil = Image.fromarray(restored_image)
        output_filename = f"{output_dir}/{image_file.split('.')[0]}_restored_Q{Q}.jpg"
        restored_image_pil.save(output_filename)
        print(f"Saved {output_filename}")

    return {"rmse": image_rmse, "bpp": image_bpp}


def plot_grouped_results(image_files, all_rmse_bpp, group_size, output_dir):
    """
    Plot RMSE vs. BPP results for groups of images.

    Args:
        image_files (list): List of image filenames.
        all_rmse_bpp (dict): RMSE and BPP data for all images.
        group_size (int): Number of images per plot.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_groups = len(image_files) // group_size
    colors = ['b', 'g', 'r', 'c', 'm']

    for group in range(num_groups):
        plt.figure(figsize=(10, 6))
        group_images = image_files[group * group_size:(group + 1) * group_size]

        for idx, image_file in enumerate(group_images):
            if image_file in all_rmse_bpp:
                rmse_values = all_rmse_bpp[image_file]["rmse"]
                bpp_values = all_rmse_bpp[image_file]["bpp"]
                plt.plot(bpp_values, rmse_values, label=image_file, color=colors[idx % len(colors)], marker='o')

        plt.title(f"RMSE vs. BPP (Images {group * group_size + 1} to {(group + 1) * group_size})")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("Root Mean Square Error (RMSE)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plot_filename = f"{output_dir}/rmse_vs_bpp_group_{group + 1}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")


def main():
    """
    Main function to process images and generate results.
    """
    image_files = get_image_files()
    Q_values = [1, 3, 5, 10, 15, 25, 40, 50, 75, 90, 100]
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_rmse_bpp = {}

    for image_file in image_files:
        print(f"Processing {image_file}...")
        padded_matrix, original_image = load_and_pad_image(image_file)
        if padded_matrix is None or original_image is None:
            continue

        all_rmse_bpp[image_file] = process_image(image_file, padded_matrix, original_image, Q_values, output_dir)

    plot_grouped_results(image_files, all_rmse_bpp, group_size=5, output_dir=output_dir)
    print("All images processed and graphs saved.")


if __name__ == "__main__":
    main()