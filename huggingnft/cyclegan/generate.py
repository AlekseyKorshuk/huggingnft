import argparse
import os
import numpy as np
import itertools
from pathlib import Path
import datetime
import time
import sys

from PIL import Image

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader

from huggingnft.cyclegan.cyclegan import GeneratorResNet, Discriminator
from huggingnft.cyclegan.cyclegan import weights_init_normal

from huggingnft.cyclegan.utils import ReplayBuffer, LambdaLR

from datasets import load_dataset

from accelerate import Accelerator
import torch.nn as nn
import torch
from huggingface_hub.hf_api import HfApi
import json
from torchvision import transforms as T
from huggan.pytorch.cyclegan.modeling_cyclegan import GeneratorResNet
from huggingnft.lightweight_gan.train import timestamped_filename
from huggingnft.lightweight_gan.lightweight_gan import Generator, LightweightGAN, evaluate_in_chunks, Trainer
from huggingface_hub import hf_hub_download, file_download
import math


def is_square(i: int) -> bool:
    return i == math.isqrt(i) ** 2
        
def load_lightweight_model(model_name, accelerator):
    file_path = file_download.hf_hub_download(
        repo_id=model_name,
        filename="config.json"
    )
    config = json.loads(open(file_path).read())
    organization_name, name = model_name.split("/")
    model = Trainer(**config, organization_name=organization_name, name=name)
    model.load(use_cpu=accelerator.device=='cpu')
    model.accelerator = accelerator
    model.GAN.to(accelerator.device)
    return model
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def generate(model_name, choice, num_steps, accelerator, args, imgs_per_tile=16, pairs=True):
    
    assert is_square(imgs_per_tile), 'Error, the number of images per tile must be a square number in order to have proper formatting of the image tiles'
    nrows = int(math.sqrt(imgs_per_tile))
    n_channels = 3
    image_size = 256
    
    # setup and transforms
    input_shape = (image_size, image_size)
    pipe_transform = T.Resize(input_shape)
    transform = Compose([
         T.ToPILImage(),
            T.Resize(input_shape),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    out_transform = T.ToPILImage()
    to_tensor = T.ToTensor()
    
    # models
    generator = load_lightweight_model(f"huggingnft/{model_name.split('__2__')[0]}", accelerator)
    translator = GeneratorResNet.from_pretrained(f'huggingnft/{model_name}', 
                    input_shape=(n_channels, image_size, image_size), 
                    num_residual_blocks=9).to(accelerator.device)
    
    
    # prediction  pipeline
    
    if choice == 'interpolate':
            res_dir, interp_collection = generator.generate_interpolation(
                num=timestamped_filename(),
                num_image_tiles=nrows,
                num_steps=num_steps,
                save_frames=False,
                progress_bar=None,
                return_preds=True,
            )            

            output_result = []

            for step, collection_at_step in enumerate(interp_collection):
                curr_step_results = []
                # pick only half of the samples from the collection, the other half is created by the respective translation
                for idx, gen_img in enumerate(collection_at_step[:(max(imgs_per_tile//2,1) if pairs else imgs_per_tile) ]):

                    input = pipe_transform(gen_img).unsqueeze(0).to(accelerator.device)
                    output = translator(input)
                    pil_out = make_grid(output.cpu(), nrow=1, normalize=True)
                    if pairs:                                            
                            pil_result = get_concat_h(
                                out_transform(make_grid(input.cpu(), nrow=1, normalize=True)), 
                                out_transform(pil_out)
                            )
                    else:
                            pil_result = out_transform(pil_out)
                    tensor_result = to_tensor(pil_result)
                    curr_step_results.append(tensor_result)

                #nrows//2 due to the fact that each image occupies two columns of a given row, given
                # the fact that these are generated as pairs (generated, translated) NFTs                                                                 
                final_output_at_step = out_transform(make_grid(curr_step_results,
                                       nrow=nrows//2 if pairs else nrows, normalize=False, padding=16 if pairs else 0))
                output_result.append(final_output_at_step)            
            
    else:
            output_result = []
            for step in range(num_steps):
                
                curr_step_results = []
                
                collection = generator.generate_app(
                        num=timestamped_filename(),
                        nrow=nrows,
                        checkpoint=-1,
                        types='default'
                )[1]
                input = pipe_transform(collection).to(accelerator.device)
                output = translator(input)
                
                for in_img, out_img in list(zip(input, output))[:(max(imgs_per_tile//2,1) if pairs else imgs_per_tile)]:
                    pil_out = make_grid(out_img.cpu(), nrow=1, normalize=True)
                    if pairs:                                            
                            pil_result = get_concat_h(
                                out_transform(make_grid(in_img.cpu(), nrow=1, normalize=True)), 
                                out_transform(pil_out)
                            )
                    else:
                            pil_result = out_transform(pil_out)
                    tensor_result = to_tensor(pil_result)
                    curr_step_results.append(tensor_result)
                
                
                final_output_at_step = out_transform(make_grid(curr_step_results,
                                       nrow=nrows//2 if pairs else nrows, 
                                        normalize=False, 
                                        padding=16 if pairs else 0))
                output_result.append(final_output_at_step)
                  
    from PIL import Image
    import glob
    save_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    offset = len(glob.glob(os.path.join(save_dir, f'*.{args.format}'  )  ) )
    
    if args.format == 'png':
        [ image.save(fp=os.path.join(save_dir, 
                                     f'{offset+idx+1}.{args.format}'
                                    ), 
                     format=args.format)\
                 for idx, image in enumerate(output_result) ]
    else:
        starting_frame = output_result[0].copy()
        fp_out = os.path.join(save_dir, f"{offset+1}.{args.format}")
        starting_frame.save(fp=fp_out, format='GIF', append_images=output_result[1:],
                 save_all=True, duration=args.gif_duration, loop=0)
            
    


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cryptopunks__2__bored-apes-yacht-club", help="name of the translation model")
    parser.add_argument(
        "--choice",
        type=str,
        default="interpolate",
        choices=["generate", "interpolate"],
        help="Whether to generate a continuous set of latent inputs that are translated by the cyclegan('interpolate' option) or simply translate a set of one or more generated examples by means of the upstream GAN ('generate' option)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["png", "gif"],
        help="Choose output format of the generated NFTs. If 'png',  each will be saved in the provided output directory, else a single .gif image will be generated with all the samples",
    )    
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output",
        help="Output directory to which the results are saved",
    )    
        
    parser.add_argument("--pairs", action="store_true", help="If passed, images will consists of the pair (generated, translated) NFTs, in which 'generated' represents the source NFT generated by the GAN, which is then translated by a CycleGAN to the 'translated' NFT.\n Otherwise, only the translations are considered")
    parser.add_argument("--num_images", type=int, default=100, help="Number of frames to include in the output GIF")
    parser.add_argument("--num_tiles", type=int, default=16, help="Number of tiles for each image frame in the output GIF")
    parser.add_argument("--gif_duration", type=int, default=200, help="Duration of the gif. Ignored if output is of type .png")

    parser.add_argument("--cpu", action="store_true", help="If passed, will predict on the CPU.")

    return parser.parse_args(args=args)


def main():
    args = parse_args()

    accelerator = Accelerator(cpu=(True if args.cpu else False ))
    generate(args.model_name, args.choice, args.num_images, accelerator, args, imgs_per_tile=args.num_tiles, pairs=(True if args.pairs else False) )

if __name__ == "__main__":
    main()
