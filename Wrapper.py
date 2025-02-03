import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#DOG FILTERS
def generate_oriented_dog_filters(scales, num_orientations):
    """Generate oriented Derivatives of Gaussian (DoG) filters."""
    filter_bank = []
    angles = np.linspace(0, 360, num_orientations, endpoint=False)

    for sigma in scales:
        k = 2 * int(4 * sigma + 0.5) + 1
        p = k // 2
        gaussian_2d = np.zeros((k, k), np.float32)

        for x in range(-p, p + 1):
            for y in range(-p, p + 1):
                gaussian_2d[x + p, y + p] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        gaussian_2d /= np.sum(gaussian_2d)

        dog_x = np.zeros_like(gaussian_2d)
        for i in range(1, gaussian_2d.shape[0] - 1):
            for j in range(1, gaussian_2d.shape[1] - 1):
                dog_x[i, j] = (
                        gaussian_2d[i - 1, j + 1] + 2 * gaussian_2d[i, j + 1] + gaussian_2d[i + 1, j + 1] -
                        gaussian_2d[i - 1, j - 1] - 2 * gaussian_2d[i, j - 1] - gaussian_2d[i + 1, j - 1]
                )

        for theta in angles:
            rotation_matrix = cv2.getRotationMatrix2D((p, p), theta, 1)
            rotated_dog = cv2.warpAffine(dog_x, rotation_matrix, (k, k), borderMode=cv2.BORDER_REPLICATE)
            filter_bank.append(rotated_dog)

    return filter_bank

#LM filter bank
def LM_filter_bank(scales, elongation=3, cols=10, output_file=None, filter_size=49, padding=2, verbose=True):
    """
    Generate Leung-Malik (LM) filter bank and save as a grid image with padding between filters."""
    num_orient = 6  # Number of orientations for derivative filters
    filter_bank = []
    angles = np.linspace(0, np.pi, num_orient, endpoint=False)  # Changed to pi instead of 360

    # Generate filters
    for sigma in scales[:3]:  # First 3 scales for derivative filters
        # Use consistent kernel size calculation
        k = int(6 * sigma)  # Changed from 4*sigma to 6*sigma for better coverage
        if k % 2 == 0:
            k += 1

        size = k // 2
        x, y = np.meshgrid(np.linspace(-size, size, k), np.linspace(-size, size, k))

        # First derivative (Gx)
        sigma_x = sigma
        sigma_y = elongation * sigma

        # Normalized first derivative
        Gx = -(x / sigma_x) * np.exp(-(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
        Gx = Gx / np.sqrt(np.sum(Gx ** 2))

        for theta in angles:
            # Convert theta to degrees for cv2.getRotationMatrix2D
            theta_deg = np.degrees(theta)
            rotation_matrix = cv2.getRotationMatrix2D((size, size), theta_deg, 1)
            rotated_Gx = cv2.warpAffine(Gx, rotation_matrix, (k, k),
                                        borderMode=cv2.BORDER_REPLICATE)
            filter_bank.append(rotated_Gx)

        # Second derivative (Gxx)
        Gxx = ((x ** 2 - sigma_x ** 2) / (sigma_x ** 4)) * np.exp(
            -(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
        Gxx = Gxx / np.sqrt(np.sum(Gxx ** 2))

        for theta in angles:
            theta_deg = np.degrees(theta)
            rotation_matrix = cv2.getRotationMatrix2D((size, size), theta_deg, 1)
            rotated_Gxx = cv2.warpAffine(Gxx, rotation_matrix, (k, k),
                                         borderMode=cv2.BORDER_REPLICATE)
            filter_bank.append(rotated_Gxx)

    # Laplacian of Gaussian (LoG) filters
    for base_sigma in scales[:4]:  # Take first 4 scales
        for multiplier in [1, 3]:  # Generate LoG at sigma and 3*sigma
            sigma = base_sigma * multiplier
            k = int(6 * sigma)
            if k % 2 == 0:
                k += 1

            size = k // 2
            x, y = np.meshgrid(np.linspace(-size, size, k), np.linspace(-size, size, k))
            r2 = x ** 2 + y ** 2

            # Normalized LoG
            log = (r2 - 2 * sigma ** 2) * np.exp(-r2 / (2 * sigma ** 2))
            log = log / np.sqrt(np.sum(log ** 2))
            filter_bank.append(log)

    # Gaussian filters
    for sigma in scales:
        k = int(6 * sigma)
        if k % 2 == 0:
            k += 1

        size = k // 2
        x, y = np.meshgrid(np.linspace(-size, size, k), np.linspace(-size, size, k))

        # Normalized Gaussian
        gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / np.sum(gaussian)
        filter_bank.append(gaussian)

    # Create grid visualization if output_file is specified
    if output_file:
        # Calculate grid dimensions
        num_filters = len(filter_bank)
        rows = int(np.ceil(num_filters / cols))

        # Create grid image
        grid_height = rows * (filter_size + padding) - padding
        grid_width = cols * (filter_size + padding) - padding
        grid_image = np.zeros((grid_height, grid_width))

        for i, filter_kernel in enumerate(filter_bank):
            r, c = i // cols, i % cols
            resized_filter = cv2.resize(filter_kernel, (filter_size, filter_size))
            normalized_filter = cv2.normalize(resized_filter, None, 0, 255, cv2.NORM_MINMAX)
            y_start = r * (filter_size + padding)
            y_end = y_start + filter_size
            x_start = c * (filter_size + padding)
            x_end = x_start + filter_size
            grid_image[y_start:y_end, x_start:x_end] = normalized_filter
        cv2.imwrite(output_file, np.uint8(grid_image))
        if verbose:
            print(f"Grid image saved to {output_file}")

    return filter_bank

#GABOR BANK
def generate_and_save_gabor_filter_bank(size=21, lambd=10.0, sigma=4.0, gamma=1.0, num_orientations=8, num_scales=5,
                                        output_file='Gabor.png'):
    def gabor_filter(size, theta, lambd, sigma, gamma):
        # Create the grid of points (x, y)
        x = np.arange(-size // 2, size // 2 + 1)
        y = np.arange(-size // 2, size // 2 + 1)
        x, y = np.meshgrid(x, y)

        # Rotate the coordinates by angle theta
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)

        # Apply the Gabor formula
        exp_term = np.exp(-(x_rot ** 2 + gamma ** 2 * y_rot ** 2) / (2 * sigma ** 2))
        cos_term = np.cos(2 * np.pi * x_rot / lambd)

        # Combine the terms to form the Gabor filter
        return exp_term * cos_term

    # Create a list to store the Gabor filters
    filters = []

    # Generate the Gabor filter bank
    for i in range(num_orientations):
        for j in range(num_scales):
            theta = i * np.pi / num_orientations  # Vary the orientation
            sigma_scale = sigma * (2 ** j)  # Vary the scale by adjusting sigma
            gabor = gabor_filter(size, theta, lambd, sigma_scale, gamma)
            filters.append(gabor)

    # Combine all the filters into a single image for saving
    rows = num_orientations
    cols = num_scales
    filter_height, filter_width = filters[0].shape

    # Create a black canvas large enough to hold all the filters
    combined_image = np.zeros((filter_height * rows, filter_width * cols), dtype=np.float32)

    for i, filter_ in enumerate(filters):
        row = i // cols
        col = i % cols
        combined_image[row * filter_height:(row + 1) * filter_height,
        col * filter_width:(col + 1) * filter_width] = filter_

    # Normalize the combined image to [0, 255] for saving
    combined_image = np.uint8(cv2.normalize(combined_image, None, 0, 255, cv2.NORM_MINMAX))

    # Save the image using OpenCV
    cv2.imwrite(output_file, combined_image)

    return filters

# def Save_dog_bank(fb, filename="DoG.png"):
#     """
#     Visualize and save the entire filter bank as an image grid.
#
#     Args:
#         filter_bank (list): List of filters (tuples of dog_x and dog_y).
#         filename (str): Filename to save the visualization.
#     """
#     num_filters = len(fb)
#     grid_size = int(np.ceil(np.sqrt(num_filters * 2)))
#     max_dim = max(max(f.shape) for f_pair in fb for f in f_pair)
#     canvas = np.zeros((grid_size * max_dim, grid_size * max_dim), dtype=np.float32)
#     for idx, (dog_x, dog_y) in enumerate(fb):
#         # Resize filters
#         resized_dog_x = cv2.resize(dog_x, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
#         resized_dog_y = cv2.resize(dog_y, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
#         # Normalize filters
#         resized_dog_x = cv2.normalize(resized_dog_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#         resized_dog_y = cv2.normalize(resized_dog_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#         # Calculate grid position
#         x = (idx % grid_size) * max_dim
#         y = (idx // grid_size) * max_dim
#         # Place filters
#         canvas[y:y + max_dim, x:x + max_dim] = resized_dog_x
#         x = ((idx + 1) % grid_size) * max_dim
#         y = ((idx + 1) // grid_size) * max_dim
#         canvas[y:y + max_dim, x:x + max_dim] = resized_dog_y
#
#     # Save the canvas as an image
#     cv2.imwrite(filename, canvas)
#     print(f"Filter bank saved as {filename}")

#Filtering image with filter bank
def compute_filter_responses(image, filter_bank):
    """Compute responses for all filters."""
    img = image.astype(np.float32) / 255.0
    height, width = img.shape
    responses = np.zeros((height, width, len(filter_bank)), dtype=np.float32)

    for i, kernel in enumerate(filter_bank):
        responses[:, :, i] = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)

    return responses
# manual code for convulution
def manual_convolution(image, kernel, border_type='reflect'):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    # Calculate padding
    pad_h = k_height // 2
    pad_w = k_width // 2

    #padding image
    if border_type == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:  # 'constant'
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output
    output = np.zeros_like(image, dtype=np.float32)

    # Perform convolution
    for y in range(i_height):
        for x in range(i_width):
            region = padded[y:y + k_height, x:x + k_width]
            output[y, x] = np.sum(region * kernel)

    return output

# coputation of texton map
def compute_texton_map(filter_responses, n_clusters=64):
    """Compute texton map using K-means clustering."""
    height, width, n_filters = filter_responses.shape
    responses_2d = filter_responses.reshape(-1, n_filters)

    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(responses_2d)

    return cluster_ids.reshape(height, width)

#generating half disk
def generate_half_disc_masks(radius, num_orientations):
    """Generate half-disc mask pairs for gradient computation."""
    masks = []
    angles = np.linspace(0, 360, num_orientations, endpoint=False)
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]

    # Create circular mask
    disc = x ** 2 + y ** 2 <= radius ** 2

    for theta in angles:
        # Convert angle to radians
        rad = np.deg2rad(theta)

        # Create line mask
        line = x * np.cos(rad) + y * np.sin(rad)

        # Create half discs
        left_mask = disc & (line <= 0)
        right_mask = disc & (line > 0)

        # Normalize masks
        left_mask = left_mask.astype(np.float32)
        right_mask = right_mask.astype(np.float32)

        if np.sum(left_mask) > 0:
            left_mask /= np.sum(left_mask)
        if np.sum(right_mask) > 0:
            right_mask /= np.sum(right_mask)

        masks.append((left_mask, right_mask))

    return masks

# def generate_half_disc_masks(radii, orientations):
#     """Generate pairs of half-disk masks."""
#     mask_pairs = []
#     mask_size = (49, 49)  # Fixed size for masks
#     center = (mask_size[0] // 2, mask_size[1] // 2)
#
#     for radius in radii:
#         for angle in orientations:
#             # Create base circle
#             y, x = np.ogrid[:mask_size[0], :mask_size[1]]
#             circle = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
#
#             # Create rotation matrix
#             theta = np.deg2rad(angle)
#             line = (x - center[1]) * np.sin(theta) - (y - center[0]) * np.cos(theta)
#
#             # Create left and right masks
#             left_mask = circle & (line <= 0)
#             right_mask = circle & (line > 0)
#
#             # Convert to float32
#             left_mask = left_mask.astype(np.float32)
#             right_mask = right_mask.astype(np.float32)
#
#             # Normalize masks
#             if np.sum(left_mask) > 0:
#                 left_mask /= np.sum(left_mask)
#             if np.sum(right_mask) > 0:
#                 right_mask /= np.sum(right_mask)
#
#             mask_pairs.append((left_mask, right_mask))
#
#     return mask_pairs

def compute_chi_square_gradient(data, mask_pairs, num_bins):
    """Compute χ² gradient using filtering approach."""

    height, width = data.shape
    gradient = np.zeros((height, width), dtype=np.float32)

    for left_mask, right_mask in mask_pairs:
        chi_sqr = np.zeros((height, width), dtype=np.float32)

        for bin_idx in range(num_bins):
            bin_mask = (data == bin_idx).astype(np.float32)

            # Compute histograms using convolution
            g = cv2.filter2D(bin_mask, -1, left_mask, borderType=cv2.BORDER_REFLECT)
            h = cv2.filter2D(bin_mask, -1, right_mask, borderType=cv2.BORDER_REFLECT)

            # Add small epsilon to avoid division by zero
            eps = 1e-10
            numerator = (g - h) ** 2
            denominator = g + h + eps

            # Update chi-square distance
            chi_sqr += numerator / denominator

        gradient = np.maximum(gradient, 0.5 * chi_sqr)

    return gradient

def compute_brightness_map(image, n_clusters=16):
    """Compute brightness map using K-means clustering."""
    pixels = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image.shape)


def compute_color_map(image, n_clusters=16):
    """Compute color map using K-means clustering in RGB space."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image.shape[:2])

def compute_gradient(map_data, mask_pairs, num_bins):
    """
    Compute gradient using manual convolution instead of filter2D.
    """
    h, w = map_data.shape
    gradient = np.zeros((h, w), dtype=np.float32)

    for left_mask, right_mask in mask_pairs:
        chi_sqr = np.zeros((h, w), dtype=np.float32)

        for bin_idx in range(num_bins):
            # Create binary mask for current bin
            bin_mask = (map_data == bin_idx).astype(np.float32)

            # Apply half-disk masks using manual convolution
            g = manual_convolution(bin_mask, left_mask)
            h = manual_convolution(bin_mask, right_mask)

            # Compute chi-square distance
            numerator = (g - h) ** 2
            denominator = g + h + 1e-10
            chi_sqr += numerator / denominator

        gradient += 0.5 * chi_sqr

    return gradient


def combine_with_baselines (combined_gradient, canny_pb, sobel_pb, w1 =0.7, w2 =0.3):
    # Ensure weights sum to 1
    assert abs(w1 + w2 - 1.0) < 1e-6, "Weights must sum to 1"

    combined_gradient_norm = cv2.normalize(combined_gradient, None, 0, 1, cv2.NORM_MINMAX)

    # # Normalize the combined gradient to [0, 1]
    # combined_gradient_norm = (combined_gradient - np.min(combined_gradient)) / (
    #             np.max(combined_gradient) - np.min(combined_gradient) + 1e-10)

    # Read and normalize baseline images if they're file paths
    # Read baseline images if they're file paths
    if isinstance(canny_pb, str):
        canny_pb = cv2.imread(canny_pb, cv2.IMREAD_GRAYSCALE)
        canny_pb = canny_pb.astype(np.float32) / 255.0

    if isinstance(sobel_pb, str):
        sobel_pb = cv2.imread(sobel_pb, cv2.IMREAD_GRAYSCALE)
        sobel_pb = sobel_pb.astype(np.float32) / 255.0

    # Combine baselines
    baseline_combination = w1 * canny_pb + w2 * sobel_pb

    # Compute final Pb-lite using element-wise multiplication
    pb_edges = combined_gradient_norm * baseline_combination


    return pb_edges

def save_filter_visualizations(filters, name, output_dir):
    n = len(filters)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    plt.figure(figsize=(20, 20))
    for i, filt in enumerate(filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(filt, cmap='gray')
        plt.axis('off')

    plt.suptitle(f'{name} Filter Bank', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name.lower()}_filters.png'))
    plt.close()

#Implementation of gradient and pb filter
def process_single_image(image_path, base_output_dir, params, canny_baseline_dir, sobel_baseline_dir):
    """Process a single image through the complete pipeline."""
    print(f"\nProcessing image: {image_path}")

    # Create output directories
    output_dirs = {
        'base': base_output_dir,
        'filters': os.path.join(base_output_dir, 'filters'),
        'lms_filters': os.path.join(base_output_dir, 'lms_filters'),
        'lml_filters': os.path.join(base_output_dir, 'lml_filters')
    }

    # Create all necessary directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Read image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Generate filter banks
    odog_filters = generate_oriented_dog_filters(params['scales'], params['num_orientations'])
    lms_filters = LM_filter_bank(
        params['scales_lms'],
        elongation=params['elongation'],
        cols=params['cols'],
        output_file=os.path.join(output_dirs['lms_filters'], "LMS_filters.png"),
        filter_size=params['filter_size'],
        padding=params['padding']
    )
    lml_filters = LM_filter_bank(
        params['scales_lml'],
        elongation=params['elongation'],
        cols=params['cols'],
        output_file=os.path.join(output_dirs['lml_filters'], "LML_filters.png"),
        filter_size=params['filter_size'],
        padding=params['padding']
    )
    gabor_filters = generate_and_save_gabor_filter_bank(
        size=params['filter_size'],
        num_orientations=params['num_orientations'],
        output_file=os.path.join(output_dirs['filters'], "Gabor.png")
    )

    # Save filter visualizations
    save_filter_visualizations(odog_filters, 'ODOG', output_dirs['base'])
    save_filter_visualizations(lms_filters, 'LMS', output_dirs['lms_filters'])
    save_filter_visualizations(lml_filters, 'LML', output_dirs['lml_filters'])
    save_filter_visualizations(gabor_filters, 'Gabor', output_dirs['filters'])

    # Compute filter responses
    print("Computing filter responses...")
    responses = compute_filter_responses(gray_image, odog_filters)

    # Generate filter banks
    odog_filters = generate_oriented_dog_filters(params['scales'], params['num_orientations'])
    lms_filters = LM_filter_bank(
        params['scales_lms'],
        elongation=params['elongation'],
        cols=params['cols'],
        output_file=os.path.join(output_dirs['lms_filters'], "LMS_filters.png"),
        filter_size=params['filter_size'],
        padding=params['padding']
    )
    lml_filters = LM_filter_bank(
        params['scales_lml'],
        elongation=params['elongation'],
        cols=params['cols'],
        output_file=os.path.join(output_dirs['lml_filters'], "LML_filters.png"),
        filter_size=params['filter_size'],
        padding=params['padding']
    )
    gabor_filters = generate_and_save_gabor_filter_bank(
        size=params['filter_size'],
        num_orientations=params['num_orientations'],
        output_file=os.path.join(output_dirs['filters'], "Gabor.png")
    )

    # Save filter visualizations
    save_filter_visualizations(odog_filters, 'ODOG', output_dirs['base'])
    save_filter_visualizations(lms_filters, 'LMS', output_dirs['lms_filters'])
    save_filter_visualizations(lml_filters, 'LML', output_dirs['lml_filters'])
    save_filter_visualizations(gabor_filters, 'Gabor', output_dirs['filters'])

    # 3. Generate texton map
    print("Computing texton map...")
    texton_map = compute_texton_map(responses, params['num_textons'])

    # 4. Compute brightness and color maps
    print("Computing brightness and color maps...")
    brightness_map = compute_brightness_map(gray_image, params['num_brightness_clusters'])
    color_map = compute_color_map(original_image, params['num_color_clusters'])

    # 5. Generate half-disc masks
    print("Generating half-disc masks...")
    mask_pairs = generate_half_disc_masks(params['gradient_radius'], params['gradient_orientations'])

    # 6. Compute gradients
    print("Computing gradients...")
    Tg = compute_chi_square_gradient(texton_map, mask_pairs, params['num_textons'])
    Bg = compute_chi_square_gradient(brightness_map, mask_pairs, params['num_brightness_clusters'])
    Cg = compute_chi_square_gradient(color_map, mask_pairs, params['num_color_clusters'])

    # Normalize and combine gradients
    Tg_norm = cv2.normalize(Tg, None, 0, 1, cv2.NORM_MINMAX)
    Bg_norm = cv2.normalize(Bg, None, 0, 1, cv2.NORM_MINMAX)
    Cg_norm = cv2.normalize(Cg, None, 0, 1, cv2.NORM_MINMAX)
    combined_gradient = (Tg_norm + Bg_norm + Cg_norm) / 3.0

    # Load baseline edge maps
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    canny_path = os.path.join(canny_baseline_dir, f"{base_name}.png")
    sobel_path = os.path.join(sobel_baseline_dir, f"{base_name}.png")


    if not os.path.exists(canny_path) or not os.path.exists(sobel_path):
        print(f"Warning: Baseline files not found for {base_name}")
        return False

        # Compute final Pb-lite output
    pb_edges = combine_with_baselines(combined_gradient, canny_path, sobel_path)




    # 7. Save results
    def save_normalized(data, name , apply_green_colormap=False):
        normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Apply green colormap if required
        if apply_green_colormap:
            # Create a green colormap
            green_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
            #green_colormap[:, 0, 1] = np.arange(256)  # Set green channel to values from 0 to 255
            # Option 1: Mix green with white (add other channels)
            green_colormap[:, 0, 0] = np.arange(256) // 2  # Blue channel
            green_colormap[:, 0, 1] = np.arange(256)  # Green channel
            green_colormap[:, 0, 2] = np.arange(256) // 2  # Red channel

            # Alternative Option 2: Reduce the intensity of green
            # green_colormap[:, 0, 1] = np.arange(256) // 2  # Half intensity green

            # Apply the colormap
            normalized_colormap = cv2.applyColorMap(normalized, green_colormap)
            normalized = normalized_colormap
        output_path = os.path.join(output_dirs['base'], f"{base_name}_{name}.png")
        cv2.imwrite(output_path, normalized)
        return output_path

    # Save individual results
    save_normalized(texton_map, "texton_map" , apply_green_colormap=True)
    save_normalized(brightness_map, "brightness_map", apply_green_colormap=True)
    save_normalized(color_map, "color_map", apply_green_colormap=True)
    save_normalized(Tg, "texture_gradient",  apply_green_colormap=True)
    save_normalized(Bg, "brightness_gradient",  apply_green_colormap=True)
    save_normalized(Cg, "color_gradient",  apply_green_colormap=True)
    save_normalized(combined_gradient, "combined_gradient")
    save_normalized(pb_edges, "pb_lite")
    save_normalized(Tg, "texture_gradient", True)
    save_normalized(Bg, "brightness_gradient", True)
    save_normalized(Cg, "color_gradient", True)
    save_normalized(combined_gradient, "combined_gradient")
    save_normalized(pb_edges, "pb_lite")


    print(f"Processing completed for {base_name}")
    return True



def main():
    # Configuration parameters
    params = {

        'scales': [1, 2, 4, 8],  # ODOG scales
        'num_orientations': 16,  # Number of orientations for ODOG
        'scales_lms': [1, np.sqrt(2), 2, 2 * np.sqrt(2)],
        'scales_lml': [np.sqrt(2), 2, 2 * np.sqrt(2), 4],
        'cols': 10,
        'elongation': 3,
        'filter_size': 49,
        'padding': 2,
        'half_disk_radius': [4, 8],
        'half_disk_orientations': [0, 45, 90, 135, 180, 225, 270, 315],
        'num_textons': 64,  # Number of texton clusters
        'num_brightness_clusters': 16,  # Number of brightness clusters
        'num_color_clusters': 16,
        'gradient_radius': 4,
        'gradient_orientations': 45
        # 'gradient_radius': 5,  # Radius for half-disc masks
        # 'gradient_orientations': 8  # Number of orientations for gradient computation
    }

    # Directories
    input_dir = "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/BSDS500/Images"
    base_output_dir = "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/phaseoutput"
    canny_baseline_dir = "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/BSDS500/CannyBaseline"
    sobel_baseline_dir = "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/BSDS500/SobelBaseline"

    output_dir = {
        'base': "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/phaseoutput",
        'filters': "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/phaseoutput/filters",
        'lms_filters': "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/phaseoutput/lms_filters",
        'lml_filters': "D:/Computer vision/Final_submission/schauhan_hw0 - Copy/Phase1/Code/phaseoutput/lml_filters"
    }

    # Create all directories
    for dir_path in output_dir.values():
        os.makedirs(dir_path, exist_ok=True)

        # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            try:
                process_single_image(image_path, base_output_dir, params, canny_baseline_dir, sobel_baseline_dir)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()


