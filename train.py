import math
import os
import sys
import argparse
import time
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.nn.utils.clip_grad import clip_grad_norm_

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import (
    load_checkpoint,
    seed_everything,
    init_distributed_device,
    is_primary,
    AverageMeter,
    str2bool,
    save_checkpoint,
)
from dataset import ImageDataset, create_transforms


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpm.yaml",
        help="config file used to specify parameters",
    )

    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/local/dataset/imagenet100_128x128/train",
        help="data folder",
    )
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_classes", type=int, default=100, help="number of classes in dataset"
    )

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="output folder"
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=5000,
        help="number of linear warmup steps before cosine decay",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.01,
        help="minimum LR as a fraction of the base LR after cosine decay",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="none",
        choices=["fp16", "bf16", "fp32", "none"],
        help="mixed precision",
    )

    # generating image count
    parser.add_argument(
        "--num_gen_images", type=int, default=16, help="number of images to generate for evaluation"
    )

    parser.add_argument("--min_snr_gamma", type=float, default=2.0, help="gamma for SNR-based loss weighting")

    parser.add_argument("--low_t_prob", type=float, default=0.0)
    parser.add_argument("--low_t_max", type=int, default=200)
    parser.add_argument("--low_t_aux_weight", type=float, default=0.2)

    # ddpm
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=1000, help="ddpm inference timesteps"
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.0002, help="ddpm beta start"
    )
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="ddpm beta schedule"
    )
    parser.add_argument(
        "--variance_type", type=str, default="fixed_small", help="ddpm variance type"
    )
    parser.add_argument(
        "--prediction_type", type=str, default="epsilon", help="ddpm epsilon type"
    )
    parser.add_argument(
        "--clip_sample",
        type=str2bool,
        default=True,
        help="whether to clip sample at each step of reverse process",
    )
    parser.add_argument(
        "--clip_sample_range", type=float, default=1.0, help="clip sample range"
    )
    parser.add_argument(
        "--image_in_size", type=int, default=128, help="cropped input image size"
    )

    # unet
    parser.add_argument(
        "--unet_in_size", type=int, default=64, help="unet input image size"
    )
    parser.add_argument(
        "--unet_in_ch", type=int, default=3, help="unet input channel size"
    )
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument(
        "--unet_ch_mult",
        type=int,
        default=[1, 2, 2, 2],
        nargs="+",
        help="unet channel multiplier",
    )
    parser.add_argument(
        "--unet_attn",
        type=int,
        default=[1, 2, 3],
        nargs="+",
        help="unet attantion stage index",
    )
    parser.add_argument(
        "--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks"
    )
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")

    # vae
    parser.add_argument(
        "--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm"
    )

    # cfg
    parser.add_argument(
        "--use_cfg",
        type=str2bool,
        default=False,
        help="use cfg for conditional (latent) ddpm",
    )
    parser.add_argument(
        "--cfg_guidance_scale", type=float, default=1.0, help="cfg for inference"
    )

    # ddim sampler for inference
    parser.add_argument(
        "--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference"
    )

    # checkpoint path for inference
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint path for inference"
    )

    # first parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args


def main():
    # wandb login
    wandb.login(
        key="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"
    )

    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup distributed initialize and device
    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    # setup dataset
    logger.info("Creating dataset")
    # use transform to normalize your images to [-1, 1]
    transform = create_transforms(image_size=args.image_in_size, augment=True)

    # use image folder for your train dataset
    train_dataset = ImageDataset(args.data_dir, transform=transform, preload=False)

    # setup dataloader
    sampler = None
    if args.distributed:
        # distributed sampler
        sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    # shuffle
    shuffle = False if sampler else True
    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    # setup experiment folder
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_name is None:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}"
    else:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}-{args.run_name}"
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, "checkpoints")
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch,
    )
    # print number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # ddpm shceduler
    scheduler = DDPMScheduler(
        args.num_train_timesteps,
        args.num_inference_steps,
        args.beta_start,
        args.beta_end,
        args.beta_schedule,
        args.variance_type,
        args.prediction_type,
        args.clip_sample,
        args.clip_sample_range,
    )

    # NOTE: this is for latent DDPM
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt("/local/pretrained/model.ckpt")
        vae.eval()

    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch, n_classes=args.num_classes
        )

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # setup optimizer
    params = list(unet.parameters())
    if class_embedder is not None:
        params += list(class_embedder.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # # setup scheduler
    # training_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.num_epochs
    # )

    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # setup scheduler: linear warmup -> cosine decay (step-based)
    def lr_lambda(current_step: int):
        warmup_steps = max(0, args.lr_warmup_steps)
        min_lr_ratio = max(0.0, min(1.0, args.min_lr_ratio))

        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        if args.max_train_steps <= warmup_steps:
            return 1.0

        progress = float(current_step - warmup_steps) / float(
            max(1, args.max_train_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    training_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )

    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[args.device],
            output_device=args.device,
            find_unused_parameters=False,
        )
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder,
                device_ids=[args.device],
                output_device=args.device,
                find_unused_parameters=False,
            )
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae
    # setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
            args.num_train_timesteps,
            args.num_inference_steps,
            args.beta_start,
            args.beta_end,
            args.beta_schedule,
            args.variance_type,
            args.prediction_type,
            args.clip_sample,
            args.clip_sample_range,
        )
    else:
        scheduler_wo_ddp = scheduler

    # setup evaluation pipeline
    # this pipeline is not differentiable and only for evaluation
    pipeline = DDPMPipeline(unet, scheduler_wo_ddp, vae, class_embedder)

    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)

    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(project="ddpm", name=args.run_name, config=vars(args))

    # Start training
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    # reload from checkpoint
    load_checkpoint(unet, scheduler_wo_ddp, vae=vae, class_embedder=class_embedder, checkpoint_path="experiments/exp-2-ddpm-cfg/checkpoints/checkpoint_epoch_151.pth")

    generator = torch.Generator(device=device)
    # ts = int(time.time())
    # same seed for image checking horizontally across epochs
    generator.manual_seed(args.seed)
    classes = torch.randint(0, args.num_classes, (args.num_gen_images,), generator=generator, device=device)
    
    # training
    for epoch in range(args.num_epochs):

        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)  # type: ignore

        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")

        loss_m = AverageMeter()
        bucket_0_199 = AverageMeter()
        bucket_200_599 = AverageMeter()
        bucket_600_999 = AverageMeter()

        # set unet and scheduelr to train
        unet.train()

        for step, (images, labels) in enumerate(train_loader):

            batch_size = images.size(0)

            # send to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # NOTE: this is for latent DDPM
            if vae is not None:
                # use vae to encode images as latents
                images = vae.encode(images)
                # do not change this line, this is to ensure the latent has unit std
                images = images * 0.1845

            # one-time sanity check: verify UNet input shape matches config
            if epoch == 0 and step == 0:
                b, c, h, w = images.shape
                logger.info(
                    f"[Sanity] UNet input tensor shape after preprocessing: {images.shape}"
                )
                if c != args.unet_in_ch:
                    raise ValueError(
                        f"Channel mismatch: tensor C={c}, but args.unet_in_ch={args.unet_in_ch}. "
                        f"Fix config or preprocessing/VAE output channels."
                    )
                if (h, w) != (args.unet_in_size, args.unet_in_size):
                    raise ValueError(
                        f"Spatial mismatch: tensor HxW=({h}, {w}), but args.unet_in_size={args.unet_in_size}. "
                        f"Fix config or preprocessing/VAE downsampling."
                    )

            # zero grad optimizer
            optimizer.zero_grad()

            # NOTE: this is for CFG
            if class_embedder is not None:
                # use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # set class_emb to None
                class_emb = None

            # sample noise
            noise = torch.randn_like(images)

            # sample timestep t
            if args.low_t_prob > 0:
                # base: uniform over all timesteps
                timesteps_full = torch.randint(
                    low=0,
                    high=args.num_train_timesteps,
                    size=(batch_size,),
                    device=device,
                    dtype=torch.long,
                )

                # low-t candidates
                timesteps_low = torch.randint(
                    low=0,
                    high=args.low_t_max,
                    size=(batch_size,),
                    device=device,
                    dtype=torch.long,
                )

                # choose which samples use low-t
                choose_low = torch.rand(batch_size, device=device) < args.low_t_prob

                timesteps = torch.where(choose_low, timesteps_low, timesteps_full)
            else:
                timesteps = torch.randint(
                    low=0,
                    high=args.num_train_timesteps,
                    size=(batch_size,),
                    device=device,
                    dtype=torch.long,
                )

            # add noise to images using scheduler
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # model prediction
            model_pred = unet(noisy_images, timesteps, class_emb)

            if args.prediction_type == "epsilon":
                target = noise
            elif args.prediction_type == "v_prediction":
                alphas_cumprod = scheduler.alphas_cumprod.to(
                    device=images.device, dtype=images.dtype
                )
                alpha_bar_t = alphas_cumprod[timesteps]

                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

                while len(sqrt_alpha_bar_t.shape) < len(images.shape):
                    sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
                    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)

                # v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x0
                target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * images
            else:
                raise ValueError(f"Unsupported prediction_type: {args.prediction_type}")

            # per-sample loss
            per_sample_loss = F.mse_loss(model_pred, target, reduction="none")
            per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))  # [B]

            alphas_cumprod = scheduler.alphas_cumprod.to(
                device=images.device, dtype=images.dtype
            )
            alpha_bar_t = alphas_cumprod[timesteps]
            snr = alpha_bar_t / (torch.clamp(1.0 - alpha_bar_t, min=1e-8))

            if args.prediction_type == "epsilon":
                weights = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma)) / snr
            elif args.prediction_type == "v_prediction":
                weights = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma)) / (snr + 1.0)
            else:
                raise ValueError(f"Unsupported prediction_type: {args.prediction_type}")
            # weights = torch.ones_like(per_sample_loss)

            weighted_per_sample_loss = weights * per_sample_loss
            loss = weighted_per_sample_loss.mean()

            low_aux_mask = timesteps < args.low_t_max

            if low_aux_mask.any():
                low_aux_loss = per_sample_loss[low_aux_mask].mean()
                loss = loss + args.low_t_aux_weight * low_aux_loss

            # total loss for backward
            # loss = per_sample_loss.mean()
            loss_m.update(loss.item(), batch_size)

            # timestep bucket logging
            mask_0_199 = (timesteps >= 0) & (timesteps <= 199)
            mask_200_599 = (timesteps >= 200) & (timesteps <= 599)
            mask_600_999 = (timesteps >= 600) & (timesteps <= 999)

            if mask_0_199.any():
                bucket_0_199.update(
                    per_sample_loss[mask_0_199].mean().item(),
                    int(mask_0_199.sum().item())
                )

            if mask_200_599.any():
                bucket_200_599.update(
                    per_sample_loss[mask_200_599].mean().item(),
                    int(mask_200_599.sum().item())
                )

            if mask_600_999.any():
                bucket_600_999.update(
                    per_sample_loss[mask_600_999].mean().item(),
                    int(mask_600_999.sum().item())
                )

            # backward
            loss.backward()

            # grad clip
            # if args.grad_clip:
            #     clip_grad_norm_(unet.parameters(), args.grad_clip)
            #     if class_embedder is not None:
            #         clip_grad_norm_(class_embedder.parameters(), args.grad_clip)

            # step your optimizer
            optimizer.step()
            training_scheduler.step() # per step

            progress_bar.update(1)

            # loss logging
            if step % 100 == 0 and is_primary(args):
                global_step = epoch * num_update_steps_per_epoch + step

                log_dict = {
                    "train/loss": loss_m.avg,
                    "train/lr": training_scheduler.get_last_lr()[0],
                    "train_bucket/raw_0_199": bucket_0_199.avg,
                    "train_bucket/raw_200_599": bucket_200_599.avg,
                    "train_bucket/raw_600_999": bucket_600_999.avg,
                }

                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, "
                    f"Loss {loss.item():.6f} ({loss_m.avg:.6f}), "
                    f"raw [0-199] {bucket_0_199.avg:.6f}, "
                    f"raw [200-599] {bucket_200_599.avg:.6f}, "
                    f"raw [600-999] {bucket_600_999.avg:.6f}"
                )

                low_mask = (timesteps >= 0) & (timesteps <= 199)
                if low_mask.any():
                    low_pred = model_pred[low_mask]
                    low_target = target[low_mask]
                    low_cos = F.cosine_similarity(
                        low_pred.reshape(low_pred.size(0), -1),
                        low_target.reshape(low_target.size(0), -1),
                        dim=1
                    ).mean().item()

                    log_dict.update({
                        "train_diag/low_pred_std": low_pred.std().item(),
                        "train_diag/low_target_std": low_target.std().item(),
                        "train_diag/low_cos": low_cos,
                    })

                    logger.info(
                        f"low-t pred std={low_pred.std().item():.4f}, "
                        f"target std={low_target.std().item():.4f}, "
                        f"cos={low_cos:.4f}"
                    )

                high_mask = (timesteps >= 600) & (timesteps <= 999)
                if high_mask.any():
                    high_pred = model_pred[high_mask]
                    high_target = target[high_mask]
                    high_cos = F.cosine_similarity(
                        high_pred.reshape(high_pred.size(0), -1),
                        high_target.reshape(high_target.size(0), -1),
                        dim=1
                    ).mean().item()

                    log_dict.update({
                        "train_diag/high_pred_std": high_pred.std().item(),
                        "train_diag/high_target_std": high_target.std().item(),
                        "train_diag/high_cos": high_cos,
                    })

                    logger.info(
                        f"high-t pred std={high_pred.std().item():.4f}, "
                        f"target std={high_target.std().item():.4f}, "
                        f"cos={high_cos:.4f}"
                    )

                wandb_logger.log(log_dict, step=global_step)


        # validation
        # send unet to evaluation mode
        unet.eval()

        # NOTE: this is for CFG
        if args.use_cfg:
            gen_images = pipeline(
                batch_size=args.num_gen_images,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
            )
        else:
            gen_images = pipeline(
                args.num_gen_images,
                args.num_inference_steps,
                None,
                args.cfg_guidance_scale,
                generator,
            )

        # create a blank canvas for the grid
        width_count = 8
        height_count = args.num_gen_images // width_count
        grid_image = Image.new("RGB", (width_count * args.image_size, height_count * args.image_size))
        # paste images into the grid
        for i, image in enumerate(gen_images):
            x = (i % width_count) * args.image_size
            y = (i // width_count) * args.image_size
            grid_image.paste(image, (x, y))

        # Send to wandb
        if is_primary(args):
            wandb_logger.log({"gen_images": wandb.Image(grid_image)})

        # save checkpoint
        if is_primary(args):
            save_checkpoint(
                unet_wo_ddp,
                scheduler_wo_ddp,
                vae_wo_ddp,
                class_embedder_wo_ddp,
                optimizer,
                epoch,
                save_dir=save_dir,
            )


if __name__ == "__main__":
    main()
