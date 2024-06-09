import os
from attacks.adversarial_attack import AttackMethod, EnabledAttackMethod
from service.service import Adverseal
from util.util import load_target_image
import gradio as gr
from PIL import Image


class AdversealApp:
    def __init__(self, processor):
        self.processor = processor

    def create_interface(self):
        default_target_image = "Seals"
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
                    alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.11, label="Alpha")
                    eps_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.062, label="Epsilon")
                    num_steps_slider = gr.Slider(minimum=1, maximum=150, step=1, value=20, label="Number of Steps")

                    attack_type_radio.change(self.toggle_sliders, inputs=attack_type_radio,
                                             outputs=[num_steps_slider, eps_slider])

                    with gr.Accordion("Advanced", open=False):
                        target_image_radio = gr.Radio(["Seals", "Fayum"], label="Select Target Image",
                                                      value=default_target_image)
                        target_image_preview = gr.Image(label="Target Image Preview", width=100,
                                                        value=load_target_image(default_target_image))
                        prompt = gr.Textbox(label="Image prompt", value="a picture")

                        target_image_radio.change(load_target_image, inputs=target_image_radio,
                                                  outputs=target_image_preview)

                    process_image_button = gr.Button("Process Image")
                    process_directory_button = gr.Button("Process Directory")

                with gr.Column(scale=3):
                    output_images = gr.Gallery(type="pil", label="Output", preview=True, height="90vh")

                process_image_button.click(
                    self.process_image_interface,
                    inputs=[input_image, target_image_radio, attack_type_radio, prompt, num_steps_slider, alpha_slider,
                            eps_slider],
                    outputs=output_images
                )
                process_directory_button.click(
                    self.process_directory_interface,
                    inputs=[input_directory, output_directory, target_image_radio, attack_type_radio, prompt, num_steps_slider,
                            alpha_slider, eps_slider],
                    outputs=output_images
                )
        return main_interface

    def toggle_sliders(self, attack_type):
        if attack_type == "PGD":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    def process_image_interface(self, uploaded_image, target_image_name, attack_type, prompt, num_steps, alpha, eps):
        # Load the target image as a PIL image
        target_image = load_target_image(target_image_name)
        attack_method = AttackMethod[attack_type]

        # Process the image and return the result
        output_img_pil = processor.process_image(uploaded_image, target_image, attack_method, prompt, num_steps, alpha, eps)
        return [output_img_pil]

    def process_directory_interface(self, input_dir, output_dir, target_image_name, attack_type, prompt, num_steps, alpha, eps):
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
            output_img_pil = processor.process_image(input_img, target_image, attack_method, prompt, num_steps, alpha, eps)

            # Add the image to the list
            output_images.append(output_img_pil)

            # Save the output image
            output_img_path = os.path.join(output_dir, filename)
            output_img_pil.save(output_img_path)
        return output_images


if __name__ == '__main__':
    processor = Adverseal()
    app = AdversealApp(processor)
    main_interface = app.create_interface()
    main_interface.launch(share=True)
