import torch
from torch import autograd
from attacks.common import encode_image, add_noise_to_latents, compute_loss
import torch.nn.functional as F


def fgsm_attack(models, input_image, accelerator, target_tensor, weight_dtype=torch.float16, instance_prompt='a picture', alpha=2.0/255.0):
    """Return new perturbed data for a single image using FGSM attack"""

    print('Performing FGSM on image with shape {}'.format(input_image.shape))
    print('Target image tensor shape: {}'.format(target_tensor.shape))
    print('Alpha: {}'.format(alpha))

    # Extract component models
    tokenizer = models['tokenizer']
    text_encoder = models['text_encoder']
    unet = models['unet']
    noise_scheduler = models['scheduler']
    vae = models['vae']

    device = accelerator.device

    # Move models to the specified device and set data types
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Disable gradient computation
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Create two copies of the original image
    perturbed_image = input_image.detach().clone().requires_grad_(True)
    original_image = perturbed_image.clone()
    print('Made two copies of the input image with shape {}'.format(original_image.shape))

    # Tokenize input prompt
    input_ids = tokenizer(
        instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    print('Tokenized input prompt: {}'.format(input_ids))

    # Encode the current perturbed image
    perturbed_image.requires_grad_(False)
    latents = encode_image(vae, perturbed_image, device, weight_dtype)
    latents.requires_grad_(True)
    print('Encoded the perturbed image. Shape: {}'.format(perturbed_image.cpu().detach().numpy().shape))

    # Generate noise to be added *to the latents*
    noise = torch.randn_like(latents)
    print('Generated noise for the latent according to standard distribution')

    batch_size = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
    print('Generated timestep: {}'.format(timesteps[0]))

    # Add noise to the current perturbed image latents
    noisy_latents = add_noise_to_latents(noise_scheduler, latents, noise, timesteps)

    # Encode the tokenized text prompt
    encoder_hidden_states = text_encoder(input_ids)[0]
    print('Encoded text prompt shape: {}'.format(encoder_hidden_states.cpu().detach().numpy().shape))

    # UNet predicts the current noise
    model_prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    print('Prediction shape: {}'.format(model_prediction.cpu().detach().numpy().shape))

    # Loss is the MSE between the predicted noise and the generated noise
    loss = compute_loss(noise_scheduler, model_prediction, latents, noise, timesteps)

    # Targeted attack:
    # Loss is the opposite of the MSE between the predicted noise and the target tensor
    if target_tensor is not None:
        loss = -F.mse_loss(model_prediction, target_tensor)

    # Compute the gradients
    grads = autograd.grad(loss, latents)[0].detach().clone()
    print('Gradients shape: {}'.format(grads.cpu().numpy().shape))

    # Nudge the image according to the gradients (FGSM step)
    perturbed_image.requires_grad_(True)
    gc_latents = vae.encode(perturbed_image.to(device, dtype=weight_dtype)).latent_dist.mean
    gc_latents.backward(gradient=grads)

    adv_images = perturbed_image + alpha * perturbed_image.grad.sign()

    # Ensure the image is in the color space
    perturbed_image = torch.clamp(adv_images, min=0, max=1).detach_().requires_grad_(True)

    # Display the perturbed image
    # display_image(perturbed_image, 0, 1)

    print(f"FGSM loss: {loss.detach().item()}")

    return perturbed_image.detach()
