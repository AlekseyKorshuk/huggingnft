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
from huggingnft.cyclegan.cyclegan import TEMPLATE_CYCLEGAN_CARD_PATH
import torch.nn as nn
import torch

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--source_dataset_name", type=str, default="Chris1/cryptopunks_HQ", help="name of the source dataset")
    parser.add_argument("--target_dataset_name", type=str, default="Chris1/bored_apes_yacht_club_HQ", help="name of the target dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU threads to use during batch generation")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images for training")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
            "--push_to_hub",
            action="store_true",
            help="Whether to push the model to the HuggingFace hub after training.",
            )
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")

    parser.add_argument(
        "--organization_name",
        required="--push_to_hub" in sys.argv,
        type=str,
        help="Organization name to push to, in case args.push_to_hub is specified.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Name of the directory to dump generated images and models separately, during training.")
    return parser.parse_args(args=args)




def training_function(config, args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision)
    
    
        
    STORAGE_DIR = f"{args.source_dataset_name.split('/')[-1]}__2__{args.target_dataset_name.split('/')[-1]}"   
    OUTPUT_DIR = os.path.join(args.output_dir, STORAGE_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)    
    
    if args.wandb and accelerator.is_local_main_process:
        import wandb
        wandb.init(project=STORAGE_DIR,
                       entity="chris1nexus")    
        wandb.config = vars(args)
             
    
    # Create sample and checkpoint directories
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "saved_models"), exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    input_shape = (args.channels, args.image_size, args.image_size)
    # Calculate output shape of image discriminator (PatchGAN)
    output_shape = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(args.channels)
    D_B = Discriminator(args.channels)

    if args.epoch != 0:
        # Load pretrained models

        G_AB.load_state_dict(torch.load(
            os.path.join(OUTPUT_DIR,"saved_models/G_AB_%d.pth" % (args.epoch))  ))
        G_BA.load_state_dict(torch.load(
            os.path.join(OUTPUT_DIR,"saved_models/G_BA_%d.pth" % (args.epoch))  ))
        D_A.load_state_dict(torch.load(
            os.path.join(OUTPUT_DIR,"saved_models/D_A_%d.pth" % (args.epoch))   ))
        D_B.load_state_dict(torch.load(
            os.path.join(OUTPUT_DIR,"saved_models/D_B_%d.pth" % (args.epoch))   ))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transform = Compose([
        Resize(int(args.image_size * 1.12), Image.BICUBIC),
        RandomCrop((args.image_size, args.image_size)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def transforms(examples):
        examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    def load_and_transform(dataset_name, batch_size, num_workers):
        dataset = load_dataset(dataset_name)
        transformed_dataset = dataset.with_transform(transforms)
        transformed_dataset = transformed_dataset['train']
        splits = transformed_dataset\
        .remove_columns( [col for col in transformed_dataset.column_names if col != 'image'] )\
        .train_test_split(test_size=0.1)
        train_ds = splits['train']
        val_ds = splits['test']

        dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)
        val_dataloader = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=1, drop_last=True)    
        return dataloader, val_dataloader


    from collections import OrderedDict
    dataset_names = {'source':args.source_dataset_name, 
                     'target':args.target_dataset_name}

    loaders = {'train':{},
              'val':{}}
    for domain_id, dataset_name in dataset_names.items():
        train_loader, val_loader = load_and_transform(dataset_name, args.batch_size, args.num_workers)
        loaders['train'][domain_id] = train_loader
        loaders['val'][domain_id] = val_loader



    def sample_images(args, batches_done):
        """Saves a generated sample from the test set"""
        real_A = next(iter(loaders["val"]["source"]))['image']
        real_B = next(iter(loaders["val"]["target"]))['image']
        G_AB.eval()
        G_BA.eval()
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1).to('cpu')
        save_path = os.path.join(OUTPUT_DIR,  "images/%s.png" % (batches_done) )
        return image_grid, save_path



    G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, loaders['train']['source'], loaders['train']['target'], loaders['val']['source'], loaders['val']['target']  = accelerator.prepare(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, loaders['train']['source'], loaders['train']['target'], loaders['val']['source'], loaders['val']['target'] )
    print('Starting training')
    # ----------
    #  Training
    # ----------
    SOURCE_LOADER_BATCHES = len(loaders['train']['source'])
    TARGET_LOADER_BATCHES = len(loaders['train']['target'])
    LOADER_BATCHES = min(SOURCE_LOADER_BATCHES, TARGET_LOADER_BATCHES)
    prev_time = time.time()
    for epoch in range(args.epoch, args.num_epochs):
        for i, (source_batch, target_batch) in enumerate(zip(loaders['train']['source'], loaders['train']['target']) ):

            # Set model input
            real_A = source_batch['image'] 
            real_B = target_batch['image'] 
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *output_shape), device=accelerator.device)
            fake = torch.zeros((real_A.size(0), *output_shape), device=accelerator.device)
            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity
    
            accelerator.backward(loss_G)
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            accelerator.backward(loss_D_A)
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            accelerator.backward(loss_D_B)
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------


            #'''
            if accelerator.state.num_processes > 1:

                    errD = accelerator.gather(loss_D).sum() / accelerator.state.num_processes
                    errG = accelerator.gather(loss_G).sum() / accelerator.state.num_processes
                    errGAN = accelerator.gather(loss_GAN).sum() / accelerator.state.num_processes
                    errCycle = accelerator.gather(loss_cycle).sum() / accelerator.state.num_processes   
                    errIdentity = accelerator.gather(loss_identity).sum() / accelerator.state.num_processes


                    err_id_A = accelerator.gather(loss_id_A).sum() / accelerator.state.num_processes
                    err_id_B = accelerator.gather(loss_id_B).sum() / accelerator.state.num_processes
                    err_cycle_A = accelerator.gather(loss_cycle_A).sum() / accelerator.state.num_processes
                    err_cycle_B = accelerator.gather(loss_cycle_B).sum() / accelerator.state.num_processes   
                    err_D_A = accelerator.gather(loss_D_A).sum() / accelerator.state.num_processes
                    err_D_B = accelerator.gather(loss_D_B).sum() / accelerator.state.num_processes
                    err_G_A = accelerator.gather(loss_GAN_AB).sum() / accelerator.state.num_processes
                    err_G_B = accelerator.gather(loss_GAN_BA).sum() / accelerator.state.num_processes                    
            else:

                    errD = loss_D.item()
                    errG = loss_G.item()
                    errGAN = loss_GAN.item()
                    errCycle = loss_cycle.item()  
                    errIdentity = loss_identity.item()


                    err_id_A = loss_id_A.item()
                    err_id_B = loss_id_B.item()
                    err_cycle_A = loss_cycle_A.item()
                    err_cycle_B = loss_cycle_B.item()  
                    err_D_A = loss_D_A.item()
                    err_D_B = loss_D_B.item()
                    err_G_A = loss_GAN_AB.item()
                    err_G_B = loss_GAN_BA.item() 
                    
            # Determine approximate time left
            batches_done = i + epoch * LOADER_BATCHES 
            batches_left = args.num_epochs * LOADER_BATCHES - batches_done 
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()                    
            if accelerator.is_local_main_process:
                    # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                        % (
                            epoch,
                            args.num_epochs,
                            i,
                            LOADER_BATCHES,
                            errD,
                            errG,
                            errGAN,
                            errCycle,
                            errIdentity,
                            time_left,
                        )
                    )
                    train_logs = {
                                "epoch": epoch,
                                "total_loss": errG,
                                "loss_discriminator": errD,
                                "loss_generator": errGAN,
                                "loss_cycle": errCycle,
                                "loss_identity": errIdentity,

                                "loss_identity_A": err_id_A,
                                "loss_identity_B": err_id_B,

                                "loss_cycle_A": err_cycle_A,
                                "loss_cycle_B": err_cycle_B,

                                "loss_D_A": err_D_A,
                                "loss_D_B": err_D_B,

                                "loss_G_A": err_G_A,
                                "loss_G_B": err_G_B,                    

                            }

                    if args.wandb :
                           wandb.log(train_logs)
                    # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                    image_grid, save_path = sample_images(args, batches_done)
                    if args.wandb and  accelerator.is_local_main_process:
                        try:
                            save_image(image_grid, save_path, normalize=False)
                            images = wandb.Image(str(save_path) )
                            wandb.log({'generated_examples': images }  )
                        except Exception as e:
                            print(e)
                            pass
 

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            G_AB_path = os.path.join(OUTPUT_DIR, "saved_models/G_AB_%d.pth" % (epoch))
            G_BA_path = os.path.join(OUTPUT_DIR,"saved_models/G_BA_%d.pth" % (epoch))
            D_A_path = os.path.join(OUTPUT_DIR,"saved_models/D_A_%d.pth" % (epoch))
            D_B_path = os.path.join(OUTPUT_DIR,"saved_models/D_B_%d.pth" % (epoch))

            def unwrap_and_save(model, save_filepath):
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), save_filepath)


            if accelerator.state.num_processes > 1:
                # wait all processes before proceeding with the save, as in multi GPU training some processes can still be training while the main starts saving.
                accelerator.wait_for_everyone()
            unwrap_and_save(G_AB, G_AB_path)
            unwrap_and_save(G_BA, G_BA_path)
            unwrap_and_save(D_A, D_A_path)
            unwrap_and_save(D_B, D_B_path)


    # Optionally push to hub
    if accelerator.is_main_process and args.push_to_hub:

        d1 = args.source_dataset_name.split('/')[-1]
        d2 = args.target_dataset_name.split('/')[-1]
        direct = f"{d1}__2__{d2}"
        reverse = f"{d2}__2__{d1}"  

        
        if accelerator.state.num_processes > 1:
            # wait all processes before proceeding with the save, as in multi GPU training some processes can still be training while the main starts saving.
            accelerator.wait_for_everyone()


        
        accelerator.unwrap_model(G_AB).push_to_hub(
            repo_path_or_name=f'{direct}',
            organization=args.organization_name,
            default_model_card=TEMPLATE_CYCLEGAN_CARD_PATH
        )
        
        accelerator.unwrap_model(G_BA).push_to_hub(
            repo_path_or_name=f'{reverse}',
            organization=args.organization_name,
            default_model_card=TEMPLATE_CYCLEGAN_CARD_PATH
        )        
        

def main():
    args = parse_args()
    print(args)


    training_function({}, args)


if __name__ == "__main__":
    main()
