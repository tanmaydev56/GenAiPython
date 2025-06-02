# creating a image geenerating model using pure python (using google colab)

# step -1 
# !pip install diffusers transformers accelerate safetensors tensorflow

# step -2 
from diffusers import StableDiffusionPipeline
from google.colab import files
import torch

# Initialize the pipeline with optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # Faster on GPU
    variant="fp16",
    use_safetensors=True  # Safer model loading
).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate an image
prompt = "A cyberpunk city at night, neon lights, 4k"
image = pipe(
    prompt,
    num_inference_steps=30,  # Balances speed/quality
    guidance_scale=7.5  # How closely to follow the prompt
).images[0]
display(image)

image.save("generated_image.png")


# do this on the google colab




