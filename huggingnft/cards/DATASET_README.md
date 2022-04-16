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

# Dataset Card

## Disclaimer

All rights belong to their owners.
Models and datasets can be removed from the site at the request of the copyright holder.

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [How to use](#how-to-use)
- [Dataset Structure](#dataset-structure)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
- [About](#about)

## Dataset Description

- **Homepage:** [https://github.com/AlekseyKorshuk/huggingnft](https://github.com/AlekseyKorshuk/huggingnft)
- **Repository:** [https://github.com/AlekseyKorshuk/huggingnft](https://github.com/AlekseyKorshuk/huggingnft)
- **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)


### Dataset Summary

NFT images dataset for unconditional generation.

NFT collection available [here](https://opensea.io/collection/USERNAME).

Model is available [here](https://huggingface.co/huggingnft/USERNAME).

Check Space: [link](https://huggingface.co/spaces/AlekseyKorshuk/huggingnft).

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)


## How to use

How to load this dataset directly with the datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("huggingnft/USERNAME")
```

## Dataset Structure

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)


### Data Fields

The data fields are the same among all splits.

- `image`: an `image` feature.
- `id`: an `int` feature.
- `token_metadata`: a `str` feature.
- `image_original_url`: a `str` feature.

### Data Splits

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)


## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Citation Information

```
@InProceedings{huggingnft,
    author={Aleksey Korshuk}
    year=2022
}
```


## About

*Built by Aleksey Korshuk*

[![Follow](https://img.shields.io/github/followers/AlekseyKorshuk?style=social)](https://github.com/AlekseyKorshuk)

[![Follow](https://img.shields.io/twitter/follow/alekseykorshuk?style=social)](https://twitter.com/intent/follow?screen_name=alekseykorshuk)

[![Follow](https://img.shields.io/badge/dynamic/json?color=blue&label=Telegram%20Channel&query=%24.result&url=https%3A%2F%2Fapi.telegram.org%2Fbot1929545866%3AAAFGhV-KKnegEcLiyYJxsc4zV6C-bdPEBtQ%2FgetChatMemberCount%3Fchat_id%3D-1001253621662&style=social&logo=telegram)](https://t.me/joinchat/_CQ04KjcJ-4yZTky)

For more details, visit the project repository.

[![GitHub stars](https://img.shields.io/github/stars/AlekseyKorshuk/huggingnft?style=social)](https://github.com/AlekseyKorshuk/huggingnft)
