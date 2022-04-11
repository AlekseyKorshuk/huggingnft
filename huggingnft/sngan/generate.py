import torch
from PIL import Image
from torchvision.utils import save_image
from train import Generator
from huggingface_hub import hf_hub_download


model_name = "huggingnft/cyberkongz"
model = Generator(3)
weights_path = hf_hub_download(model_name, 'pytorch_model.bin')
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with torch.no_grad():
    z = torch.randn(8, 100, 1, 1, device=device)
    pixel_values = model(z)

save_image(pixel_values, "generated.png", normalize=True)
img = Image.open(f"generated.png").convert('RGBA')
# img.putalpha(255)
img.save("generated.png")
