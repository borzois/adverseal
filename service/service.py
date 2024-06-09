import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, PNDMScheduler, AutoencoderKL
from accelerate import Accelerator
from attacks.adversarial_attack import adversarial_attack
from util.util import preprocess, tensor_to_pil
from PIL import Image
import numpy as np


class Adverseal:
    def __init__(self, model_path='models/stable-diffusion/', weight_dtype=torch.float16, max_img_size=800):
        self.weight_dtype = weight_dtype
        self.max_img_size = max_img_size
        self.cuda_enabled = torch.cuda.is_available()

        self.accelerator = Accelerator(mixed_precision="fp16", cpu=not self.cuda_enabled)

        self.models = self._load_models(model_path)

    def _load_models(self, model_path):
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder')
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet')
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer')
        scheduler = PNDMScheduler.from_pretrained(model_path, subfolder='scheduler')

        vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
        if self.cuda_enabled:
            vae = vae.cuda()
        vae.to(self.accelerator.device, dtype=self.weight_dtype)

        return {
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'unet': unet,
            'scheduler': scheduler,
            'vae': vae
        }

    def process_image(self, input_img, target_img, attack_method, prompt, num_steps, alpha, eps):
        input_img = self._resize_image(input_img)
        input_img_tensor = preprocess(input_img).unsqueeze(0)

        target_img = target_img.convert("RGB").resize(input_img.size)
        target_img = np.array(target_img)[None].transpose(0, 3, 1, 2)

        if self.cuda_enabled:
            input_img_tensor = input_img_tensor.to("cuda", dtype=self.weight_dtype)
            target_img_tensor = torch.from_numpy(target_img).to("cuda", dtype=self.weight_dtype) / 255.0
        else:
            input_img_tensor = input_img_tensor.to(dtype=self.weight_dtype)
            target_img_tensor = torch.from_numpy(target_img).to(dtype=self.weight_dtype) / 255.0

        target_latent_tensor = self.models['vae'].encode(target_img_tensor).latent_dist.sample().to(
                dtype=self.weight_dtype) * self.models['vae'].config.scaling_factor

        target_img_tensor = target_img_tensor.to('cpu')
        del target_img_tensor
        for model in self.models.values():
            if model is not None and type(model) != CLIPTokenizer and type(model) != PNDMScheduler:
                model.to('cpu')

        adversarial_image = adversarial_attack(self.models, attack_method, input_img_tensor, self.accelerator,
                                               target_latent_tensor, instance_prompt=prompt, num_steps=num_steps, alpha=alpha, eps=eps)
        return tensor_to_pil(adversarial_image)

    def _resize_image(self, img):
        width, height = img.size
        if width > self.max_img_size or height > self.max_img_size:
            scaling_factor = min(self.max_img_size / width, self.max_img_size / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            return img.resize((new_width, new_height), Image.BICUBIC)
        return img

