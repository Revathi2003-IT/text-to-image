import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

pipe = pipe.to(device)

# Prompt
prompt = "A beautiful sunset over a lake with mountains, ultra realistic, 4k"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")

# Display image
plt.imshow(image)
plt.axis("off")
plt.show()
