import matplotlib
# matplotlib.use('Agg')
import time
import numpy as np
# import cupy as cp
import matplotlib.pyplot as plt
import imageio
import io
import pydicom
from tqdm import tqdm
from IPython.display import Image as IPImage, display
import streamlit as st

# ==== FUNCTIONS ====

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)

def save_plot_to_memory(original, individual_grating, rec_grating, rec_image, idx):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    original_display = normalize_image(np.asarray(original.real))
    ig_display = normalize_image(np.asarray(individual_grating.real))
    rg_display = normalize_image(np.asarray(rec_grating.real))
    r_display = normalize_image(np.asarray(rec_image.real))

    axes[0].imshow(np.clip(original_display, 0, 1), cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    axes[1].imshow(np.clip(ig_display, 0, 1), cmap='gray')
    axes[1].axis("off")
    axes[1].set_title("Current Individual Grating")
    
    axes[2].imshow(np.clip(rg_display, 0, 1), cmap='gray')
    axes[2].axis("off")
    axes[2].set_title("Current Transformed Grating")

    axes[3].imshow(np.clip(r_display, 0, 1), cmap='gray')
    axes[3].axis("off")
    axes[3].set_title("Reconstructed Image")

    for ax in axes:
        ax.axis('off')

    fig.suptitle(f"Terms: {idx}", fontsize=16)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf    

def live_fourier_reconstruction(image_array, ft, coords_list, centre_x, centre_y, max_iter=1000):
    st.write("Starting reconstruction...")
    rec_image = np.zeros_like(image_array)
    individual_grating = np.zeros_like(image_array, dtype=np.complex64)

    progress_bar = st.progress(0)
    image_placeholder = st.empty()

    display_all_until = 100
    save_interval = 2000

    idx = 0
    saved_frames = []

    for coords in tqdm(coords_list):
        if idx >= max_iter:
            break

        if not (coords[1] == centre_y and coords[0] > centre_x):
            idx += 1
            symm_coords = find_symmetric_coordinates(coords, centre_x, centre_y)

            # Update individual_grating
            individual_grating[coords[0], coords[1]] = ft[coords[0], coords[1]]
            individual_grating[symm_coords[0], symm_coords[1]] = ft[symm_coords[0], symm_coords[1]]

            # Compute reconstruction
            rec_grating = calculate_2dift(individual_grating)
            rec_image += rec_grating

            # Save frames at key points
            if idx <= display_all_until or (idx > display_all_until and idx % save_interval == 0):
                buf = save_plot_to_memory(image_array, individual_grating, rec_grating, rec_image, idx)
                image_placeholder.image(buf, caption=f"Iteration {idx}", use_container_width=True)

                saved_frames.append(imageio.v2.imread(buf))

                progress_bar.progress(min(idx / max_iter, 1.0))
                time.sleep(0.01)

            # Clear for next iteration
            individual_grating[coords[0], coords[1]] = 0
            individual_grating[symm_coords[0], symm_coords[1]] = 0

    progress_bar.empty()
    st.success("Reconstruction completed!")

   # Save the final image buffer to session_state
    final_buf = save_plot_to_memory(image_array, individual_grating, rec_grating, rec_image, idx)
    final_bytes = final_buf.getvalue()
    saved_frames.append(imageio.v2.imread(io.BytesIO(final_bytes)))  # Safe read
    st.session_state['final_reconstructed_image'] = final_bytes  # Safe store
    
    gif_path = "fourier_reconstruction.gif"
    imageio.mimsave(gif_path, saved_frames, duration=0.1)
    st.image(gif_path, caption="Fourier Reconstruction GIF", use_container_width=True)

    # Optional: Download button for GIF
    with open(gif_path, "rb") as f:
        st.download_button("Download Reconstruction GIF", f, file_name="fourier_reconstruction.gif")

    return rec_image


def calculate_distance_from_centre(coords, centre_x, centre_y):
    x, y = coords
    return np.sqrt((x - centre_x)**2 + (y - centre_y)**2)

def find_symmetric_coordinates(coords, centre_x, centre_y):
    x, y = coords
    return (2*centre_x - x, 2*centre_y - y)

def calculate_2dft(channel_data):
    ft = np.fft.ifftshift(channel_data)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(channel_data):
    ift = np.fft.ifftshift(channel_data)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

# ==== MAIN APP ====

st.title("Fourier Reconstruction")

# Load your image
import PIL.Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    image_array = np.asarray(image, dtype=np.complex64)
    # image_array = np.array(im).astype(np.float32)

    # Prepare Fourier transform
    ft = calculate_2dft(image_array)
    # Compute magnitude spectrum for visualization
    ft_magnitude = np.log(np.abs(ft) + 1e-8)


    # Generate coordinate list
    height, width = image_array.shape
    centre_x = (height - 1) // 2
    centre_y = (width - 1) // 2

    coords_left_half = ((x, y) for x in range(height) for y in range(centre_y + 1))
    coords_left_half = sorted(
        coords_left_half,
        key=lambda c: calculate_distance_from_centre(c, centre_x, centre_y)
    )

    # Initialize session state
    if "reconstruction_images" not in st.session_state:
        st.write("Running reconstruction...")
        st.session_state.reconstruction_images = live_fourier_reconstruction(
            image_array, ft, coords_left_half, centre_x, centre_y, max_iter=len(coords_left_half)
        )

    # Always display final image if available
    if 'final_reconstructed_image' in st.session_state:
        st.image(st.session_state['final_reconstructed_image'], caption="Final Reconstructed Image", use_container_width=True)

else:
    st.info("Please upload an image to start the reconstruction.")
