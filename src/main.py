r"""Barycentric Guidance: valence-arousal editing with foundation image editors.

Example (10x10 VA grids on AffectNet with FLUX.2 [dev], barycentric guidance + face morph):
    python src/main.py --model flux2-dev --mode bary_guide --morph --scale_w_cfg --smoothness2d \
        --img_dir /path/to/affectnet/Manually_Annotated_Images --affectnet_csv /path/to/affectnet/validation.csv \
        --output_dir /path/to/outputs --cache_dir /path/to/hf_cache

See README.md for more examples.
"""
import argparse
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from dataset import make_dataset
from guidance import MODES, edit_va
from pipelines import MODELS, load_editor
from utils import DEFAULT_MESH_CSV, load_va_mesh, polar_to_cartesian, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default='flux2-dev', help="Foundation image editor to use.")
    # Data / IO
    parser.add_argument("--img_dir", required=True, help="Directory of source face images.")
    parser.add_argument("--affectnet_csv", default=None, help="AffectNet annotation CSV. If given, --img_dir is treated as AffectNet and filtered by it.")
    parser.add_argument("--output_dir", required=True, help="Directory where generated images are saved.")
    parser.add_argument("--cache_dir", required=True, help="Directory where the diffusion model is saved.")
    parser.add_argument("--mesh_csv", default=DEFAULT_MESH_CSV, help="CSV with columns Emotion,Pleasure_Mean,Arousal_Mean defining the VA mesh anchors (default: src/va_mesh.csv).")
    # Experiments
    parser.add_argument('--fid', action='store_true', help="Generate images at random VA targets (FID measurement).")
    parser.add_argument('--smoothness2d', action='store_true', help="Generate a 10x10 VA grid per source image (smoothness2d measurement).")
    parser.add_argument("--total_images", default=100, type=int, help="Number of source images for --fid / --smoothness2d.")
    # Method
    parser.add_argument("--mode", choices=MODES, default='bary_guide', help="'bary_guide': barycentric guidance (ours); 'prompt_interp': prompt embedding interpolation.")
    parser.add_argument('--morph', action='store_true', help="Use face morphing guidance.")
    parser.add_argument("--guidance_scale", default=4.0, type=float, help="Guidance scale for both the anchor images and the final edit (true_cfg_scale for Qwen).")
    parser.add_argument('--scale_w_cfg', action='store_true', help="Scale the guidance scale with the radius of the VA point.")
    parser.add_argument("--morph_step", default=10, type=int, help="Number of early steps that use the morphed image.")
    parser.add_argument("--anchor_steps", default=30, type=int, help="Denoising steps for the anchor emotion images.")
    parser.add_argument("--inference_steps", default=30, type=int, help="Denoising steps for the final images.")
    return parser.parse_args()

def va_path(save_dir, base, p):
    return os.path.join(save_dir, f"{base}_{p[0]:.2f}_{p[1]:.2f}.jpg")

def run_fid(args, generate):
    """One image per source at a VA target sampled uniformly in the unit disc (polar)."""
    save_dir = os.path.join(args.output_dir, "fid")
    os.makedirs(save_dir, exist_ok=True)

    # Resume: source basenames already processed (output format: {base}_{v:.2f}_{a:.2f}.jpg)
    suffix_re = re.compile(r'^(.+)_-?\d+\.\d{2}_-?\d+\.\d{2}\.jpg$')
    processed = {m.group(1) for f in os.listdir(save_dir) if (m := suffix_re.match(f))}
    remaining = args.total_images - len(processed)
    print(f"FID: {len(processed)} source images already processed, generating {max(remaining, 0)} more.")

    dataset = make_dataset(args.img_dir, args.affectnet_csv)
    generated = 0
    for i in tqdm(range(len(dataset))):
        if generated >= remaining:
            break
        base = dataset.files[i].split('.')[0]
        if base in processed:
            continue
        source_image, _ = dataset[i]

        angle, mag = np.random.rand() * 2 * np.pi, np.random.rand()
        va = np.array(polar_to_cartesian(mag, angle))

        img, _ = generate(source_image, va, seed=42)
        img.save(va_path(save_dir, base, va))
        generated += 1


def run_smoothness2d(args, generate):
    """10x10 VA grid over [-1, 1]^2 for each of the first `total_images` (neutral) sources."""
    save_dir = os.path.join(args.output_dir, "smoothness2d")
    os.makedirs(save_dir, exist_ok=True)
    grid = np.mgrid[-1.0:1.0:10j, -1.0:1.0:10j].reshape(2, -1).T

    dataset = make_dataset(args.img_dir, args.affectnet_csv, neutral_only=True)
    for i in tqdm(range(min(args.total_images, len(dataset)))):
        source_image, filename = dataset[i]
        base = filename.split('.')[0]
        emotion_cache = {}  # anchor images are reused across the grid for this source
        for p in grid:
            img_path = va_path(save_dir, base, p)
            if os.path.exists(img_path):
                continue
            try:
                img, emotion_cache = generate(source_image, p, seed=i, emotion_cache=emotion_cache)
                img.save(img_path)
            except Exception as e:
                print(f"Skipping {base} VA=({p[0]:.2f},{p[1]:.2f}): {e}", flush=True)

@torch.no_grad()
def main():
    set_seed(0)
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    editor = load_editor(args.model, args.cache_dir, device)
    _, points, tri, emotion_prompts = load_va_mesh(args.mesh_csv)

    def generate(source_image, va, seed, emotion_cache=None):
        return edit_va(editor, source_image, va, points, tri, emotion_prompts, args,
                       seed=seed, emotion_cache=emotion_cache)

    if args.fid:
        run_fid(args, generate)
    if args.smoothness2d:
        run_smoothness2d(args, generate)

if __name__ == '__main__':
    main()
