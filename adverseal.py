import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DiffusionPipeline, PNDMScheduler, AutoencoderKL
from accelerate import Accelerator
from enum import Enum
from attacks.adversarial_attack import adversarial_attack, AttackMethod
from util.util import *
import gradio as gr


class EnabledAttackMethod(Enum):
    PGD = AttackMethod.PGD.value
    FGSM = AttackMethod.FGSM.value


default_target_image = "Seals"

WEIGHT_DTYPE = torch.float16
CUDA_ENABLED = torch.cuda.is_available()
STABLE_DIFFUSION_PATH = 'models/stable-diffusion/'

if not CUDA_ENABLED:
    accelerator = Accelerator(
        mixed_precision="fp16",
        cpu=True
    )
else:
    accelerator = Accelerator(
        mixed_precision="fp16"
    )

# model = DiffusionPipeline.from_pretrained(STABLE_DIFFUSION_PATH, use_safetensors=True)
text_encoder = CLIPTextModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='text_encoder')
unet = UNet2DConditionModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='unet')
tokenizer = CLIPTokenizer.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='tokenizer')
scheduler = PNDMScheduler.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='scheduler')

vae = None
if not CUDA_ENABLED:
    vae = AutoencoderKL.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='vae').cuda()
else:
    vae = AutoencoderKL.from_pretrained(STABLE_DIFFUSION_PATH, subfolder='vae')
vae.to(accelerator.device, dtype=WEIGHT_DTYPE)

models = {
    'tokenizer': tokenizer,
    'text_encoder': text_encoder,
    'unet': unet,
    'scheduler': scheduler,
    'vae': vae
}


# process one image
def process_image(input_img, target_img, attack_method, num_steps, alpha, eps):
    # Convert the input PIL images to tensors
    input_img_tensor = preprocess(input_img).unsqueeze(0)

    # Scale the target image to match the input image
    target_img = target_img.convert("RGB").resize(input_img.size)
    target_img = np.array(target_img)[None].transpose(0, 3, 1, 2)

    if CUDA_ENABLED:
        input_img_tensor = input_img_tensor.to("cuda", dtype=WEIGHT_DTYPE)
        target_img_tensor = torch.from_numpy(target_img).to("cuda", dtype=WEIGHT_DTYPE) / 255.0
    else:
        input_img_tensor = input_img_tensor.to(dtype=WEIGHT_DTYPE)
        target_img_tensor = torch.from_numpy(target_img).to(dtype=WEIGHT_DTYPE) / 255.0

    print_tensor_info()
    # Encode the target image to get the latent tensor
    target_latent_tensor = vae.encode(target_img_tensor).latent_dist.sample().to(dtype=WEIGHT_DTYPE) * vae.config.scaling_factor

    # Free the memory
    target_img_tensor = target_img_tensor.to('cpu')
    del target_img_tensor
    for model in models.values():
        if model is not None:
            model.to('cpu')

    # Perform the adversarial attack
    adversarial_image = adversarial_attack(models, attack_method, input_img_tensor, accelerator, target_latent_tensor, num_steps=num_steps, alpha=alpha, eps=eps)

    # Convert the adversarial image tensor to a PIL image
    adversarial_img_pil = tensor_to_pil(adversarial_image)

    return adversarial_img_pil


# INTERFACES
# process one image
def process_image_interface(uploaded_image, target_image_name, attack_type, num_steps, alpha, eps):
    # Load the target image as a PIL image
    target_image = load_target_image(target_image_name)

    attack_method = AttackMethod[attack_type]

    # Process the image and return the result
    output_img_pil = process_image(uploaded_image, target_image, attack_method, num_steps, alpha, eps)

    return [output_img_pil]


# process a directory
def process_directory_interface(input_dir, output_dir, target_image_name, attack_type, num_steps, alpha, eps):
    # Load the target image as a PIL image
    target_image = load_target_image(target_image_name)

    attack_method = AttackMethod[attack_type]

    os.makedirs(output_dir, exist_ok=True)

    output_images = []
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue  # Skip non-image files

        # Load the image
        input_img_path = os.path.join(input_dir, filename)
        input_img = Image.open(input_img_path).convert("RGB")

        # Process the image
        output_img_pil = process_image(input_img, target_image, attack_method, num_steps, alpha, eps)

        # Add the image to the list
        output_images.append(output_img_pil)

        # Save the output image
        output_img_path = os.path.join(output_dir, filename)
        output_img_pil.save(output_img_path)

    return output_images


def toggle_sliders(attack_type):
    if attack_type == "PGD":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


with gr.Blocks(theme="gradio/monochrome", title="Adverseal") as main_interface:
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Image"):
                input_image = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])
            with gr.Tab("Batch"):
                input_directory = gr.Textbox(label="Input Directory")
                output_directory = gr.Textbox(label="Output Directory")

            attack_type_radio = gr.Radio([method.name for method in EnabledAttackMethod], label="Attack Type",
                                         value="PGD")
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.018, label="Alpha")
            eps_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.062, label="Epsilon")
            num_steps_slider = gr.Slider(minimum=1, maximum=150, step=1, value=20, label="Number of Steps")

            attack_type_radio.change(
                toggle_sliders,
                inputs=attack_type_radio,
                outputs=[num_steps_slider, eps_slider]
            )

            with gr.Accordion("Advanced", open=False):
                target_image_radio = gr.Radio(["Seals", "Fayum"], label="Select Target Image",
                                              value=default_target_image)
                target_image_preview = gr.Image(label="Target Image Preview", width=100,
                                                value=load_target_image(default_target_image))

                target_image_radio.change(
                    load_target_image,
                    inputs=target_image_radio,
                    outputs=target_image_preview
                )

            process_image_button = gr.Button("Process Image")
            process_directory_button = gr.Button("Process Directory")

        with gr.Column(scale=3):
            output_images = gr.Gallery(type="pil", label="Adversarial Image", preview=True, height="90vh")

        process_image_button.click(
            process_image_interface,
            inputs=[input_image, target_image_radio, attack_type_radio, num_steps_slider, alpha_slider, eps_slider],
            outputs=output_images
        )
        process_directory_button.click(
            process_directory_interface,
            inputs=[input_directory, output_directory, target_image_radio, attack_type_radio, num_steps_slider,
                    alpha_slider, eps_slider],
            outputs=output_images
        )


if __name__ == '__main__':
    main_interface.launch(share=True)
