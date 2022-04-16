# Hugging NFT

<img src="https://raw.githubusercontent.com/AlekseyKorshuk/huggingnft/main/docs/banner.png" alt="Banner" width="1200"/>

---

**Hugging NFT** — generate NFT or train new model in just few clicks! Train as much as you can, others will resume from
checkpoint!

# How to generate

## Images and Interpolation

### Space

You can easily use Space: [link](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft)

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/generate.ipynb)

Follow this
link: [link](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/generate.ipynb)

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
generation_type = "default"  # ["default", "ema"]

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

## How to train

You can easily add new model for any OpenSea collection. Note that it is important to collect dataset before training —
check the corresponding section.

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/train.ipynb)

Follow this
link: [link](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/train.ipynb)

### Terminal

You can now run script as follows:

```bash
accelerate config
```

=> Accelerate will ask what kind of environment you'd like to run your script on, simply answer the questions being
asked. Next:

```bash
accelerate launch huggingnft/lightweight_gan/train.py \
  --wandb \
  --image_size 256 \
  --num_train_steps 10000 \
  --save_every 1000 \
  --dataset_name huggingnft/cyberkongz \
  --push_to_hub \
  --name cyberkongz \
  --organization_name huggingnft
```

## Collection2Collection

TODO

# Collect dataset

Because OpenSea usually blocks any api connection, we are going to use Selenium to parse data. So first
download `chromedriver` from [here](https://chromedriver.chromium.org/downloads) and pass corresponding path:

```bash
python huggingnft/datasets/collect_dataset.py --collection_name cyberkongz --use_selenium --driver_path huggingnft/datasets/chromedriver
```

# Model overfitting

There is a possibility that you can overtrain the model. In such case you can revert best commit with this
notebook: [link](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/select_best_model.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/lightweight_gan/select_best_model.ipynb)

> With great power comes great responsibility!

# About

_Built by Aleksey Korshuk, Christian Cancedda and Hugging Face community with love_ ❤️

[![Follow](https://img.shields.io/github/followers/AlekseyKorshuk?style=social)](https://github.com/AlekseyKorshuk)

[![Follow](https://img.shields.io/twitter/follow/alekseykorshuk?style=social)](https://twitter.com/intent/follow?screen_name=alekseykorshuk)

Star project repository:

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/huggingnft?style=social)](https://github.com/AlekseyKorshuk/huggingnft)


