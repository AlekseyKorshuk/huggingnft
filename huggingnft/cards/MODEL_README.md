---
tags:
- huggingnft
- nft
- huggan
- gan
- image
- images
task: 
- unconditional-image-generation
datasets:
- huggingnft/USERNAME
license: mit
---

# Hugging NFT: USERNAME

## Disclaimer

All rights belong to their owners. Models and datasets can be removed from the site at the request of the copyright
holder.

## Model description

SN-GAN model for unconditional generation.

NFT collection available [here](https://opensea.io/collection/USERNAME).

Dataset is available [here](https://huggingface.co/datasets/huggingnft/USERNAME).

Check Space: [link](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft).

Project repository: [link](https://github.com/AlekseyKorshuk/huggingnft).

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/huggingnft?style=social)](https://github.com/AlekseyKorshuk/huggingnft)

## Intended uses & limitations

#### How to use

```python
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from huggingface_hub import PyTorchModelHubMixin


class Generator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_channels=4, latent_dim=100, hidden_size=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            nn.ConvTranspose2d(hidden_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, noise):
        pixel_values = self.model(noise)

        return pixel_values


model = Generator.from_pretrained("huggingnft/USERNAME")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with torch.no_grad():
    z = torch.randn(1, 100, 1, 1, device=device)
    pixel_values = model(z)

# turn into actual image
image = pixel_values[0]
image = (image + 1) / 2
image = ToPILImage()(image)
image.save("generated.png")
```

#### Limitations and bias

Check project repository: [link](https://github.com/AlekseyKorshuk/huggingnft).

## Training data

Dataset is available [here](https://huggingface.co/datasets/huggingnft/USERNAME).

## Training procedure

Training script is available [here](https://github.com/AlekseyKorshuk/huggingnft).

## Generated Images

Check results with Space: [link](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft).

## About

*Built by Aleksey Korshuk*

[![Follow](https://img.shields.io/github/followers/AlekseyKorshuk?style=social)](https://github.com/AlekseyKorshuk)

[![Follow](https://img.shields.io/twitter/follow/alekseykorshuk?style=social)](https://twitter.com/intent/follow?screen_name=alekseykorshuk)

[![Follow](https://img.shields.io/badge/dynamic/json?color=blue&label=Telegram%20Channel&query=%24.result&url=https%3A%2F%2Fapi.telegram.org%2Fbot1929545866%3AAAFGhV-KKnegEcLiyYJxsc4zV6C-bdPEBtQ%2FgetChatMemberCount%3Fchat_id%3D-1001253621662&style=social&logo=telegram)](https://t.me/joinchat/_CQ04KjcJ-4yZTky)

For more details, visit the project repository.

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/huggingnft?style=social)](https://github.com/AlekseyKorshuk/huggingnft)

### BibTeX entry and citation info

```bibtex
@InProceedings{huggingnft,
    author={Aleksey Korshuk}
    year=2022
}
```