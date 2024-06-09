import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import gc
import torch


def print_tensor_info():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def display_image(image, step, total_steps):
    """Display the perturbed image using matplotlib."""
    img_np = image.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    print('Displaying image with size {}'.format(img_np.shape))
    plt.imshow(img_np)
    plt.title(f"Step {step + 1}/{total_steps}")
    plt.show()


def display_diff(image1, image2, scale=1.0):
    print(image1.min(), image1.max(), image2.min(), image2.max())
    # Convert PIL images to NumPy arrays if they are not already
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Ensure the images are the same size
    if image1_np.shape != image2_np.shape:
        raise ValueError("Images must have the same dimensions for comparison")

    # print(image1_np)
    # print(image2_np)
    # Compute the absolute difference between the two images
    difference_np = np.abs(image1_np.astype(np.float32) - image2_np.astype(np.float32)) * scale

    # Clamp the difference to the range [0, 1]
    difference_np = np.clip(difference_np, 0, 1)

    # print(difference_np)

    # Display the images and their difference side by side
    plt.figure(figsize=(15, 5))

    # Original image 1
    plt.subplot(1, 3, 1)
    plt.imshow(image1_np, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')

    # Original image 2
    plt.subplot(1, 3, 2)
    plt.imshow(image2_np, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    # Difference image
    plt.subplot(1, 3, 3)
    plt.imshow(difference_np, cmap='gray')
    plt.title('Perturbation')
    plt.axis('off')

    plt.show()


def tensor_to_np(img_tensor):
    img_np = img_tensor.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    return img_np


def tensor_to_pil(tensor):
    tensor = tensor.squeeze().detach().cpu()
    img = transforms.ToPILImage()(tensor)
    return img


preprocess = transforms.Compose([
    transforms.ToTensor(),
])


def load_target_image(target_image_name):
    TARGET_IMAGE_PATHS = {
        "Seals": "data/seals.png",
        "Fayum": "data/fayum.jpg"
    }
    return Image.open(TARGET_IMAGE_PATHS[target_image_name])
