import numpy as np
from dct_quantization import *
from huffman_rle import *
import pickle

def compress_image(image, quantization_matrix):
    image = image - 128  # Center the image values to range [-128, 128]
    
    # Compute DCT blocks
    dct_blocks = compute_dct_blocks(image)
    
    # Initialize lists to hold RLE-encoded blocks
    all_rle_dc = []
    all_rle_ac = []

    # Compute DC coefficients and their differences for encoding
    prev_dc = None
    dc_coeffs = []

    for block in dct_blocks:
        quantized_block = quantize_block(block, quantization_matrix)
        
        # Extract DC coefficient (top-left corner of the DCT block)
        dc = quantized_block[0, 0]
        
        if prev_dc is not None:
            # Calculate the difference between successive DC coefficients
            dc_diff = dc - prev_dc
            dc_coeffs.append(dc_diff)
        else:
            # For the first block, just use the DC coefficient as is
            dc_coeffs.append(dc)
        
        # Update previous DC for next iteration
        prev_dc = dc

        # For AC coefficients, apply RLE
        ac_block = quantized_block  # Exclude the DC coefficient
        rle_ac = run_length_encode(ac_block)  # Flatten AC coefficients and encode with RLE
        all_rle_ac.extend(rle_ac)
    # Run-Length Encode the DC coefficients
    # rle_dc = run_length_encode(np.array(dc_coeffs).reshape(1, -1))  # RLE for the DC coefficient differences
    # all_rle_dc.extend(rle_dc)
    # print("Hello")
    # print(all_rle_ac)
    # Encode the values using Huffman coding
    huffman1 = HuffmanCoding()
    huffman2 = HuffmanCoding()
    dc_values = [value for value in dc_coeffs]
    ac_values = [value for _, _, value in all_rle_ac]
    huffman_codes_dc = huffman1.encode(dc_values)  # Huffman codes for DC
    huffman_codes_ac = huffman2.encode(ac_values)  # Huffman codes for AC
    # print(huffman_codes_ac)
    
    # Ensure the Huffman codes are binary
    assert all(set(code) <= {"0", "1"} for code in huffman_codes_dc.values()), "Huffman codes for DC contain non-binary characters"
    assert all(set(code) <= {"0", "1"} for code in huffman_codes_ac.values()), "Huffman codes for AC contain non-binary characters"

    # Encode each block with Huffman codes (separate for DC and AC)
    encoded_blocks_dc = []
    encoded_blocks_ac = []
    block_sizes_dc = []
    block_sizes_ac = []
    prev_dc = None


    for block in dct_blocks:
        quantized_block = quantize_block(block, quantization_matrix)
        
        # Encode DC coefficients using Huffman
        dc = quantized_block[0, 0]
        dc_diff = dc - prev_dc if prev_dc is not None else dc
        prev_dc = dc
        # encoded_dc = huffman_encode_rle_with_global_codes(dc_diff, huffman_codes_dc)
        encoded_dc = huffman_codes_dc.get(dc_diff, None)
        # print(dc_diff)
        # print(encoded_dc)
        encoded_blocks_dc.append(encoded_dc)
        block_sizes_dc.append(len(encoded_dc))
        # Encode AC coefficients using Huffman
        ac_block = quantized_block  # Exclude DC coefficient
        rle_ac = run_length_encode(ac_block)
        encoded_ac = huffman_encode_rle_with_global_codes(rle_ac, huffman_codes_ac)
        encoded_blocks_ac.append(encoded_ac)
        block_sizes_ac.append(len(encoded_ac))

    # Encode the Huffman table into binary
    huffman_binary_map_dc = huffman1.encode_mapping()
    huffman_binary_map_ac = huffman2.encode_mapping()
    # print("Hello")
    # print(huffman_codes_ac)
    x = huffman1.decode_mapping(huffman_binary_map_dc)
    y = huffman2.decode_mapping(huffman_binary_map_ac)
    # print(y)
    # print(y[2])
    # print(huffman_codes_ac[2])
    # print(huffman_codes_ac)
    # print(diff_between_dicts(y, huffman_codes_ac))
    # print(huffman_codes_ac)
    
    assert y == huffman_codes_ac
    assert x == huffman_codes_dc
    
    # Encode the quantization matrix
    quantization_matrix_binary = "".join(
        f"{int(value):016b}" for value in quantization_matrix.flatten()
    )
    # Concatenate all binary data into a single string
    compressed_data = f"{len(encoded_blocks_ac):016b}"  # Number of blocks (AC)
    # print(len(encoded_blocks_ac))
    compressed_data += f"{image.shape[0]:016b}{image.shape[1]:016b}"  # Image dimensions (16 bits each)
    # print(image.shape[0])
    # print(image.shape[1])
    # print(len(compressed_data))
    # Huffman mapping lengths and byte arrays
    compressed_data += f"{len(huffman_binary_map_dc):032b}"  # Length of Huffman map (DC)
    # print(len(huffman_binary_map_dc)
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters-2"
    
    compressed_data += huffman_binary_map_dc  # Huffman mapping binary (DC)
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters-1"
    
    compressed_data += f"{len(huffman_binary_map_ac):032b}"  # Length of Huffman map (AC)
    # print(len(huffman_binary_map_ac))
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters0"
    
    compressed_data += huffman_binary_map_ac  # Huffman mapping binary (AC)
    # print(len(compressed_data))
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters1"
    
    for block, block_size in zip(encoded_blocks_dc, block_sizes_dc):
        compressed_data += f"{block_size:010b}"  # Size of each DC block
        compressed_data += block  # DC block binary data
    # print(len(compressed_data))
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters2"
    for block, block_size in zip(encoded_blocks_ac, block_sizes_ac):
        compressed_data += f"{block_size:016b}"  # Size of each AC block
        compressed_data += block  # AC block binary data
    # print(len(compressed_data))
    # Quantization matrix
    compressed_data += quantization_matrix_binary
    # print(len(compressed_data))
    # Ensure only `0` and `1` in the final compressed data
    assert set(compressed_data) <= {"0", "1"}, "Compressed data contains non-binary characters3"

    # Save to a binary file
    # print(len(compressed_data))
# Calculate the remainder when the length of compressed_data is divided by 8
    remainder = len(compressed_data) % 8
    # print(remainder)
    remainder_bits = f"{remainder:03b}"
    remaining_data = compressed_data 
    with open("compressed_image.txt", "wb") as file:
        # Write the first 3 bits as a single byte (padding with 0s to make it a full byte)
        file.write(int(remainder_bits, 2).to_bytes(1, byteorder="big"))
        # Write the remaining compressed data as a byte array
        file.write(int(remaining_data, 2).to_bytes((len(remaining_data) + 7) // 8, byteorder="big"))
        # print(remaining_data[0:48])
    # print(sys.getsizeof('compressed_image.txt'))
    return "compressed_image.txt"


def decode(codes, encoded_data):
    """
    Decode a binary string encoded with global Huffman codes.
    Format:
    - First 4 bits: Number of zeros (run-length) before the Huffman code.
    - Next 4 bits: Size of the Huffman code (number of bits in the code).
    - Next `size` bits: The actual Huffman code.

    Args:
        encoded_data (str): The encoded binary string.

    Returns:
        list: A 1D array representing the reconstructed sequence, including zeros.
    """
    reverse_codes = {v: k for k, v in codes.items()}  # Reverse mapping for decoding
    decoded = []  # Final result array
    index = 0  # Pointer in the encoded binary string
    j = 0
    initial_run_length = 0
    # print(len(encoded_data))
    while index+12 < len(encoded_data):
        # Read the run-length (4 bits)
        run_length = int(encoded_data[index:index + 6], 2)
        index += 6
        # print(run_length)
        # Read the size of the Huffman code (4 bits)
        size = int(encoded_data[index:index + 6], 2)
        index += 6

        # Read the Huffman code of the specified size
        huffman_code = encoded_data[index:index + size]
        index += size

        # Decode the Huffman code to find the value
        value = reverse_codes.get(huffman_code, None)
        # if value is None:
            # print(encoded_data[index-size-8:index])
            # print(decoded)
            # raise ValueError(f"Invalid Huffman code: {huffman_code}")
            # value = 0
        # print(f'{huffman_code}, {value} and {run_length} and {size}')

        # Append the zeros (based on run-length) and the value to the decoded array
        decoded.extend([0]* run_length)
        decoded.append(value)
        # print(decoded)
        j += 1
        # print(value)
        # initial_run_length = run_length
    return decoded


def decompress_image(filename):
    with open(filename, "rb") as file:
        compressed_data_file = file.read()  # Read binary data from the file
    # Extract the first byte containing the remainder
    remainder_byte = compressed_data_file[0]  # First byte
    remainder = remainder_byte & 0b111  # Extract the last 3 bits using a bitwise AND
    # Remaining binary data
    # print(remainder)
    compressed_data_bytes = compressed_data_file[1:]  # All bytes except the first one
    # Convert the remaining binary data to an integer and then to a binary string
    # print(len(compressed_data_bytes)*8)
    compressed_data = bin(int.from_bytes(compressed_data_bytes, byteorder="big"))[2:]
    # print(compressed_data[0:48])
    # Ensure the binary string length is a multiple of 8 minus the remainderif remainder == 0
    if remainder == 0:
        remainder = 8
    expected_length = (len(compressed_data_bytes)-1) * 8 + remainder  # Adjust for the remainder bits
    # print(len(compressed_data))
    compressed_data = compressed_data.zfill(expected_length)
    # print(compressed_data[0:48])

    # compressed_data = bin(int.from_bytes(compressed_data, byteorder="big"))[2:].zfill(len(compressed_data) * 8)
    index = 0
    # print(len(compressed_data))
    
    # Extract number of blocks (DC and AC)
    num_blocks_ac = int(compressed_data[index:index + 16], 2)
    index += 16
    # print(num_blocks_ac)
    # Extract image dimensions (height, width)
    height = int(compressed_data[index:index + 16], 2)
    index += 16
    # print(height)
    width = int(compressed_data[index:index + 16], 2)
    index += 16
    # print(width)
    # print(index)
    # Extract Huffman mappings (DC and AC)
    huffman_map_length_dc = int(compressed_data[index:index + 32], 2)
    index += 32
    # print(huffman_map_length_dc)
    huffman_binary_map_dc = compressed_data[index:index + huffman_map_length_dc]
    index += huffman_map_length_dc
    # print(compressed_data[0:index+32])
    huffman_map_length_ac = int(compressed_data[index:index + 32], 2)
    index += 32
    # print(huffman_map_length_ac)
    huffman_binary_map_ac = compressed_data[index:index + huffman_map_length_ac]
    index += huffman_map_length_ac
    # print(index)
    # Decode the Huffman codes for DC and AC
    huffman = HuffmanCoding()
    codes_dc = huffman.decode_mapping(huffman_binary_map_dc)
    reverse_dc_codes = {v: k for k, v in codes_dc.items()}
    codes_ac = huffman.decode_mapping(huffman_binary_map_ac)

    # Extract encoded DC and AC blocks
    encoded_blocks_dc = []
    block_sizes_dc = []
    for _ in range(height*width//64):
        block_size = int(compressed_data[index:index + 10], 2)
        block_sizes_dc.append(block_size)
        index += 10
        encoded_block = compressed_data[index:index + block_size]
        encoded_blocks_dc.append(encoded_block)
        index += block_size
    # print(index)
    encoded_blocks_ac = []
    block_sizes_ac = []
    for _ in range(height*width//64):
        block_size = int(compressed_data[index:index + 16], 2)  # Size of each AC block (10 bits)
        block_sizes_ac.append(block_size)
        index += 16
        encoded_block = compressed_data[index:index + block_size]
        encoded_blocks_ac.append(encoded_block)
        index += block_size
    # print(index)
    # Extract the quantization matrix
    quantization_matrix = np.zeros((8, 8), dtype=np.float32)
    for i in range(64):
        quantization_matrix[i // 8, i % 8] = int(compressed_data[index:index + 16], 2)
        index += 16
    # print(quantization_matrix)
    # print(index)
    # Reconstruct the image using the decoded blocks
    restored_image = np.zeros((height, width))
    block_size = 8
    prev_dc = None  # For DC coefficient differences
    # print(len(encoded_blocks_dc))
    # print(len(encoded_blocks_ac))
    # print(codes_ac)
    # Decode DC blocks and AC blocks
    for i, (encoded_dc, encoded_ac) in enumerate(zip(encoded_blocks_dc, encoded_blocks_ac)):
        row, col = divmod(i, width // block_size)
        
        # Decode DC block (using Huffman decoding and RLE)
        dc_diff = reverse_dc_codes.get(encoded_dc, None)
        dc = dc_diff + prev_dc if prev_dc is not None else dc_diff
        prev_dc = dc
        # Decode AC block (using Huffman decoding and RLE)
        ac_values = decode(codes_ac, encoded_ac)
        # Reconstruct the quantized block and dequantize it
        quantized_block = zigzag_to_block([dc]+ac_values)  # reconstruct the block from zigzag scan and DC value
        # print([dc]+ac_values)
        dequantized_block = dequantize_block(quantized_block, quantization_matrix)
        
        # Apply inverse DCT to get the final block
        restored_image[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size] = idct2(dequantized_block)
    
    restored_image = restored_image + 128  # Undo centering of image
    return np.clip(restored_image, 0, 255).astype(np.uint8)

