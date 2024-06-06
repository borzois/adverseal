def add_noise_to_latents(noise_scheduler, latents, noise, timesteps):
    """Add noise to the latents according to the noise magnitude at each timestep."""
    print("Adding noise to latents according to noise magnitude at timestep {}".format(timesteps))
    return noise_scheduler.add_noise(latents, noise, timesteps)


def compute_loss(noise_scheduler, model_prediction, latents, noise, timesteps):
    """Compute the loss based on the prediction type."""
    target = None
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # MSE between the predicted noise and the generated noise
    return F.mse_loss(model_prediction.float(), target.float(), reduction="mean")


def encode_image(vae, image, device, dtype):
    """Encode an image using the VAE to get its latent representation."""
    with torch.no_grad():
        latents = vae.encode(image.to(device, dtype=dtype)).latent_dist.mean
    return latents.detach().clone() * vae.config.scaling_factor
