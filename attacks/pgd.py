def pgd_attack(models, input_image, accelerator, target_tensor, num_steps=1, instance_prompt='a picture', gradient_accumulation_steps=1, alpha=0.005, eps=8.0/255.0):
    """Return new perturbed data for a single image"""

    step_alpha = alpha / num_steps
    print('Performing PGD on image with shape {}'.format(input_image.shape))
    print('Target image tensor shape: {}'.format(target_tensor.shape))
    print('Accumulation steps: {} | Alpha: {} | Alpha per step: {} | Epsilon: {}'.format(gradient_accumulation_steps, alpha, step_alpha, eps))

    # Extract component models
    tokenizer = models['tokenizer']
    text_encoder = models['text_encoder']
    unet = models['unet']
    noise_scheduler = models['scheduler']
    vae = models['vae']

    device = accelerator.device

    # Move models to the specified device and set data types
    vae.to(device, dtype=WEIGHT_DTYPE)
    text_encoder.to(device, dtype=WEIGHT_DTYPE)
    unet.to(device, dtype=WEIGHT_DTYPE)

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

    print('####[ Beginning optimization for {} steps ]####'.format(num_steps))
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Encode the current perturbed image
        perturbed_image.requires_grad_(False)
        latents = encode_image(vae, perturbed_image, device, WEIGHT_DTYPE)
        latents.requires_grad_(True)
        print('Encoded the perturbed image. Shape: {}'.format(perturbed_image.cpu().detach().numpy().shape))

        # Generate noise to be added *to the latents*
        # It follows standard normal distribution
        # This effectively represents a random vector in the latent space
        noise = torch.randn_like(latents)
        print('Generated noise for the latent according to standard distribution')

        batch_size = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
        print('Generated timestep: {}'.format(timesteps[0]))

        # Add noise to the current perturbed image latents
        # *in* the latent space
        # according to the random timestep
        noisy_latents = add_noise_to_latents(noise_scheduler, latents, noise, timesteps)

        # Encode the tokenized text prompt
        # Shape: 1 x num_words x dimensions
        # Shape: 1 x 77        x 768
        encoder_hidden_states = text_encoder(input_ids)[0]
        print('Encoded text prompt shape: {}'.format(encoder_hidden_states.cpu().detach().numpy().shape))

        # UNet predicts the current noise
        model_prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        print('Prediction shape: {}'.format(model_prediction.cpu().detach().numpy().shape))

        # Loss is the MSE between the predicted noise and the generated noise
        loss = compute_loss(noise_scheduler, model_prediction, latents, noise, timesteps)

        # Targeted attack:
        # Loss is the opposite of the MSE between the predicted noise and the target tensor
        # The model aims to maximize the difference between the predicted noise and the target
        if target_tensor is not None:
            loss = -F.mse_loss(model_prediction, target_tensor)

        loss /= gradient_accumulation_steps

        # Compute the gradients
        # In the latent space, this is a vector that represents
        # The direction of *maximum* growth of loss
        grads = autograd.grad(loss, latents)[0].detach().clone()
        print('Gradients shape: {}'.format(grads.cpu().numpy().shape))

        # Nudge the image according to the gradients
        # But, since we take the - of the MSE
        # We perturb the image in a way that maximizes the loss
        # This represents a SMALLER distance to the target in the latent space
        perturbed_image.requires_grad_(True)
        gc_latents = vae.encode(perturbed_image.to(device, dtype=WEIGHT_DTYPE)).latent_dist.mean
        gc_latents.backward(gradient=grads)

        # Blend the original with the perturbed
        if step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
            # FGSM?
            adv_images = perturbed_image + step_alpha * perturbed_image.grad.sign()

            # Ensure the result is in the epsilon Ball
            eta = torch.clamp(adv_images - original_image, min=-eps, max=+eps)

            # Ensure the image is in the color space
            perturbed_image = torch.clamp(original_image + eta, min=0, max=+1).detach_().requires_grad_(True)

            # Display the perturbed image every 10 steps
            # if step % 10 == 0:
                # display_image(perturbed_image, step, num_steps)

        print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
    return perturbed_image.detach()

