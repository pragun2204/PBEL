import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Automatically detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Load model with caching
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype
    )
    return pipe.to(device)

pipe = load_model()

# UI Setup
st.set_page_config(page_title="Prompt to Image Generator", layout="centered")
st.title("🎨 Prompt to Image Generator (CPU + GPU Support)")

# Prompt input
prompt = st.text_input("Enter your text prompt", "A fantasy castle on a hill")

# Generate Image
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image..."):
            result = pipe(prompt)
            image = result.images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
            image.save("generated_image.png")
            st.success("✅ Image generated and saved!")

            # Download button
            with open("generated_image.png", "rb") as file:
                st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
