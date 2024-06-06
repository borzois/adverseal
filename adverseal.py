import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DiffusionPipeline, PNDMScheduler, AutoencoderKL
from accelerate import Accelerator
from enum import Enum
from attacks.adversarial_attack import adversarial_attack
from util.util import *
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QLineEdit, QMessageBox)

# CONSTANTS
STABLE_DIFFUSION_PATH = '/content/stable-diffusion'
MAX_TRAINING_STEPS = 5
SAMPLE_FILE_PATH = 'dataset/henri.jpeg'
TARGET_IMAGE_PATH = 'dataset/MIST.png'
INPUT_DIR = '/content/drive/MyDrive/wikiart-dataset/louis-wain'
OUTPUT_DIR = '/content/output/'
WEIGHT_DTYPE = torch.float16
CUDA_ENABLED = torch.cuda.is_available()


def load_components():
    """
    Load necessary models and components.
    """
    if not CUDA_ENABLED:
        accelerator = Accelerator(mixed_precision="fp16", cpu=True)
    else:
        accelerator = Accelerator(mixed_precision="fp16")

    model = DiffusionPipeline.from_pretrained(STABLE_DIFFUSION_PATH, use_safetensors=True)
    text_encoder = CLIPTextModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='unet')
    tokenizer = CLIPTokenizer.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='tokenizer')
    scheduler = PNDMScheduler.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='scheduler')

    vae = AutoencoderKL.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='vae')
    if CUDA_ENABLED:
        vae = vae.cuda()
    vae.to(accelerator.device, dtype=WEIGHT_DTYPE)

    models = {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'unet': unet,
        'scheduler': scheduler,
        'vae': vae
    }

    return models, accelerator


def process_images(input_dir, output_dir, target_image_path, models, accelerator):
    """
    Process each image in the input directory, perform adversarial attack, and save the output.
    """
    original_target_image = Image.open(target_image_path)
    vae = models['vae']

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue  # Skip non-image files

        # Load and preprocess the image
        sample_img_path = os.path.join(input_dir, filename)
        sample_img = Image.open(sample_img_path).convert("RGB")
        sample_img_tensor = preprocess(sample_img).unsqueeze(0)  # Add batch dimension

        # Load and preprocess the target image
        target_image = original_target_image.convert("RGB").resize(sample_img.size)
        target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)

        if CUDA_ENABLED:
            sample_img_tensor = sample_img_tensor.to("cuda", dtype=WEIGHT_DTYPE)
            target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=WEIGHT_DTYPE) / 255.0
        else:
            sample_img_tensor = sample_img_tensor.to(dtype=WEIGHT_DTYPE)
            target_image_tensor = torch.from_numpy(target_image).to(dtype=WEIGHT_DTYPE) / 255.0

        target_latent_tensor = vae.encode(target_image_tensor).latent_dist.sample().to(dtype=WEIGHT_DTYPE) * vae.config.scaling_factor

        # Perform adversarial attack (assuming 'adversarial_attack' is defined elsewhere)
        adversarial_image = adversarial_attack(models, AttackMethod.FGSM, sample_img_tensor, accelerator, target_latent_tensor, num_steps=20, alpha=10.0/255.0, eps=16.0/255.0)

        # Convert the adversarial image tensor to PIL
        adversarial_img_pil = tensor_to_pil(adversarial_image)

        # Save the output image
        output_img_path = os.path.join(output_dir, filename)
        adversarial_img_pil.save(output_img_path)

        print(f"Processed and saved: {filename}")


def main():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load components
    models, accelerator = load_components()

    # Process images and save outputs
    process_images(INPUT_DIR, OUTPUT_DIR, TARGET_IMAGE_PATH, models, accelerator)

    print("All images processed.")


if __name__ == "__main__":
    main()
