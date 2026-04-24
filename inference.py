import os
import glob
import argparse
import numpy as np
import torch
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

from generate_submission import generate_submission_from_tensors

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

logger = get_logger(__name__)


# ----- output paths (edit here if needed) -----
OUTPUT_IMG_DIR = "generated_images"
SUBMISSION_CSV = "submission.csv"
COMPUTE_FID = True           # set False to skip FID (faster)
NUM_FID_REFERENCE = 5000     # subset of training data for FID reference


def main():
    args = parse_args()
    assert args.ckpt is not None, "--ckpt must be specified for inference"

    seed_everything(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # ========== build models (mirrors train.py) ==========
    logger.info("Creating model")
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
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 1e6:.2f}M")

    # scheduler (DDIM or DDPM)
    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(
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

    # VAE (frozen, pretrained)
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt("/local/pretrained/model.ckpt")
        vae.eval()

    # ClassEmbedder
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch, n_classes=args.num_classes
        )

    # to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae is not None:
        vae = vae.to(device)
    if class_embedder is not None:
        class_embedder = class_embedder.to(device)

    # ========== load checkpoint ==========
    load_checkpoint(
        unet, scheduler, vae=vae, class_embedder=class_embedder,
        checkpoint_path=args.ckpt,
    )
    unet.eval()
    if class_embedder is not None:
        class_embedder.eval()

    # ========== pipeline ==========
    pipeline = DDPMPipeline(
        unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder,
    )

    logger.info("***** Running Inference *****")

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    all_images = []   # PIL images, for saving + evaluation
    img_idx = 0

    # ========== generation loop ==========
    if args.use_cfg:
        # 50 images per class × 100 classes = 5000
        per_class = 50
        for cls_id in tqdm(range(args.num_classes), desc="class"):
            gen_images = pipeline(
                batch_size=per_class,
                num_inference_steps=args.num_inference_steps,
                classes=cls_id,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            for img in gen_images:
                img.save(os.path.join(OUTPUT_IMG_DIR, f"{img_idx:05d}.png"))
                img_idx += 1
            all_images.extend(gen_images)
    else:
        # unconditional: 5000 images in chunks of 50
        per_batch = 50
        total = 5000
        for _ in tqdm(range(0, total, per_batch), desc="batch"):
            gen_images = pipeline(
                batch_size=per_batch,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            for img in gen_images:
                img.save(os.path.join(OUTPUT_IMG_DIR, f"{img_idx:05d}.png"))
                img_idx += 1
            all_images.extend(gen_images)

    logger.info(f"Generated {len(all_images)} images → {OUTPUT_IMG_DIR}/")

    # ========== convert PIL → uint8 tensor for metrics ==========
    to_tensor = T.PILToTensor()   # uint8 (C, H, W)
    gen_tensors = torch.stack([to_tensor(img) for img in all_images])  # (N, 3, H, W) uint8
    logger.info(f"Generated tensor shape: {gen_tensors.shape}, dtype: {gen_tensors.dtype}")

    # ========== Inception Score ==========

    logger.info("Computing Inception Score...")
    is_metric = InceptionScore(feature=2048, normalize=False).to(device)
    metric_batch = 64
    for i in tqdm(range(0, len(gen_tensors), metric_batch), desc="IS update"):
        is_metric.update(gen_tensors[i:i + metric_batch].to(device))
    is_mean, is_std = is_metric.compute()
    logger.info(f"IS: {is_mean.item():.4f} ± {is_std.item():.4f}")

    # ========== FID against training subset ==========
    if COMPUTE_FID and os.path.isdir(args.data_dir):
        logger.info(f"Computing FID (reference: {NUM_FID_REFERENCE} real images from {args.data_dir})")
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

        # collect reference image paths
        patterns = ("*.JPEG", "*.jpg", "*.jpeg", "*.png")
        ref_paths = []
        for p in patterns:
            ref_paths.extend(glob.glob(os.path.join(args.data_dir, "**", p), recursive=True))
        logger.info(f"Found {len(ref_paths)} real images; sampling {NUM_FID_REFERENCE}")

        rng = np.random.default_rng(args.seed)
        ref_paths = rng.choice(
            ref_paths, size=min(NUM_FID_REFERENCE, len(ref_paths)), replace=False
        )

        ref_transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.PILToTensor(),  # uint8
        ])

        # update FID with real images
        buffer, buffer_size = [], metric_batch
        for p in tqdm(ref_paths, desc="FID real"):
            buffer.append(ref_transform(Image.open(p).convert("RGB")))
            if len(buffer) == buffer_size:
                fid_metric.update(torch.stack(buffer).to(device), real=True)
                buffer = []
        if buffer:
            fid_metric.update(torch.stack(buffer).to(device), real=True)

        # update FID with generated images
        for i in tqdm(range(0, len(gen_tensors), metric_batch), desc="FID fake"):
            fid_metric.update(gen_tensors[i:i + metric_batch].to(device), real=False)

        fid = fid_metric.compute()
        logger.info(f"FID (vs {NUM_FID_REFERENCE} real): {fid.item():.4f}")

    # ========== Kaggle submission CSV ==========

    gen_float = gen_tensors.float() / 255.0  # [0, 1]
    generate_submission_from_tensors(
        gen_float,
        output_csv=SUBMISSION_CSV,
        device=device,
    )
    logger.info(f"Submission written to {SUBMISSION_CSV}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
