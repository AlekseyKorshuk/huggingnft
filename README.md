# Hugging NFT

<img src="https://raw.githubusercontent.com/AlekseyKorshuk/huggingnft/main/docs/banner.png" alt="Banner" width="1200"/>

---

**Hugging NFT** — generate NFT or train new model in just few clicks! Train as much as you can, others will resume from checkpoint!


# How to generate

## Images and Interpolation

### Space
You can easily use Space: [link](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft)

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/generate.ipynb)

Follow this link: [link](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/generate.ipynb)

### Terminal

#### Image

```bash
python huggingnft/lightweight_gan/generate_image.py --collection_name cryptopunks --nrows 8 --generation_type default
```

#### Interpolation

```bash
python huggingnft/lightweight_gan/generate_interpolation.py --collection_name cryptopunks --nrows 8 --num_steps 100
```
### Python code

#### Image

```python
from huggingnft.lightweight_gan.train import timestamped_filename
from huggingnft.lightweight_gan.lightweight_gan import load_lightweight_model

collection_name = "cyberkongz"
nrows = 8
generation_type = "default" # ["default", "ema"]

model = load_lightweight_model(f"huggingnft/{collection_name}")
image_saved_path, generated_image = model.generate_app(
    num=timestamped_filename(),
    nrow=nrows,
    checkpoint=-1,
    types=generation_type
)
```

#### Interpolation

```python
from huggingnft.lightweight_gan.train import timestamped_filename
from huggingnft.lightweight_gan.lightweight_gan import load_lightweight_model

collection_name = "cyberkongz"
nrows = 1
num_steps = 100

model = load_lightweight_model(f"huggingnft/{collection_name}")
gif_saved_path = model.generate_interpolation(
        num=timestamped_filename(),
        num_image_tiles=nrows,
        num_steps=num_steps,
        save_frames=False
)
```

## Collection2Collection


# How to train


