# Hugging NFT

<img src="https://raw.githubusercontent.com/AlekseyKorshuk/huggingnft/main/docs/banner.png" alt="Banner" width="1200"/>

---

**Hugging NFT** — generate NFT or train new model in just few clicks! Train as much as you can, others will resume from
checkpoint!

---

![Example](https://raw.githubusercontent.com/AlekseyKorshuk/cdn/main/huggingnft/giphy.gif)

🤗 More examples are available here: [EXAMPLES.md](https://github.com/AlekseyKorshuk/huggingnft/blob/main/docs/EXAMPLES.md).

> This preview does not show the real power of this project because of a strong decrease in video quality! Otherwise, the file size would exceed all limits.

# How to generate

## Space

You can easily use Space: [link](https://huggingface.co/spaces/huggan/huggingnft)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft)

## Images and Interpolation

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
The collection2collection framework allows to create unpaired image translation models between any pair of NFT collections that can be downloaded from Opensea.   
In the broadest sense, it allows to apply the style of a collection to that of another one, so as to obtain new and diverse collections of never before seen NFTs.  


### Jupyter notebook
The training procedure is provided in a simplified format in the jupyter notebook 
[train_cyclegans.ipynb](https://github.com/AlekseyKorshuk/huggingnft/blob/main/huggingnft/cyclegan/train_cyclegans.ipynb)    

here, hyperparameter optimization is available by adding multiple parameters to each list of hyperparameters shown in the notebook.   
Furthermore, a section in such notebook is dedicated to the training of all possible translations by means of the datasets provided in the [huggingnft organization page](https://huggingface.co/huggingnft)


### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/huggingnft/blob/main/huggingnft/cyclegan/train_cyclegans.ipynb)



### Terminal
Firstly, after cloning [this repository](https://github.com/AlekseyKorshuk/huggingnft.git), run   
```bash
cd huggingnft
pip install .
```

Then, set the wandb API_KEY if you wish to log all results to wandb with:   
```bash
wandb login API_KEY
```

If you plan on uploading the resulting models to an huggingface repository, make sure to also login with your huggingface API_KEY with the following command:   
```bash
huggingface-cli login 
```

Before starting the model training, it is necessary to configure the accelerate environment according to your available computing resource with the command:  
```bash
accelerate config
```

After this, everything is setup to start training the collection2collection models

```bash
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
        train.py \
        --batch_size 8 \
        --beta1 0.5 \
        --beta2 0.999 \
        --channels 3 \ 
        --checkpoint_interval 5 \
        --decay_epoch 80 \
        --epoch 0 \
        --image_size 256 \
        --lambda_cyc 10.0 \
        --lambda_id 5.0 \ 
        --lr 0.0002 \
        --mixed_precision no \
        --model_name cyclegan \
        --n_residual_blocks 9 \
        --num_epochs 200 \
        --num_workers 8 \
        --organization_name huggingnft \
        --push_to_hub \
        --sample_interval 10 \
        --source_dataset_name huggingnft/azuki \ 
        --target_dataset_name huggingnft/boredapeyachtclub \
        --wandb \
        --output_dir experiments
```
### Generate collection2collection examples
Head to the huggingnft cyclegan subfolder and utilize the generate.py script to create NFTs with the collection2collection models at [huggingNFT](https://huggingface.co/spaces/huggan/huggingnft)

#### Generate one NFT

To generate a collection of num_images NFTs which are the outputs of the generation+translation pipeline do:
```bash
python3 generate.py --choice generate \
    --num_tiles 1 \
    --num_images 1 \
    --format png
```

####  Generate multiple NFTs 
To generate a gif containing pairs of generated and the corresponding translated NFTs (both do not exist and are the predictions of GANs), set --format png to save each image separately instead of a condensed gif
```bash
python3 generate.py \
    --choice generate \
    --num_tiles 1 \ 
    --num_images 100 \
    --format gif \
    --pairs
```

#### Visualize multiple NFTs, side by side comparison of generated and resulting translation 
To generate a gif containing pairs of generated and the corresponding translated NFTs (both do not exist and are the predictions of GANs)   this command allows to observe how contiguous changes in the latent space of the upstream GAN which generates the samples, affect the following translation by the CycleGAN  

Set --format png to save each image separately instead of a condensed gif
```bash
python3 generate.py --choice interpolate \
    --num_tiles 16 \
    --num_images 100\ 
    --format gif\
     --pairs
```
#### Visualize multiple NFTs, only the resulting translation

Remove the --pairs argument in order to visualize only the result of the translation
```bash
python3 generate.py --choice interpolate \
    --num_tiles 16 \
    --num_images 100\ 
    --format gif\
```     


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
[![Follow](https://img.shields.io/github/followers/chris1nexus?style=social)](https://github.com/Chris1nexus)

[![Follow](https://img.shields.io/twitter/follow/alekseykorshuk?style=social)](https://twitter.com/intent/follow?screen_name=alekseykorshuk)
[![Follow](https://img.shields.io/twitter/follow/chris_cancedda?style=social)](https://twitter.com/intent/follow?screen_name=chris_cancedda)

Star project repository:

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/huggingnft?style=social)](https://github.com/AlekseyKorshuk/huggingnft)


