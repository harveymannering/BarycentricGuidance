# Barycentric Guidance

[![arXiv](https://img.shields.io/badge/arXiv-2610.16892-b31b1b.svg)](https://arxiv.org/abs/2610.16892) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harveymannering/BarycentricGuidance/blob/main/notebooks/barycentric_guidance_klein.ipynb)

Code for *Barycentric Guidance: Turning Foundation Image Editors into Continuous Affective Controllers* (NeurIPS 2026).

Barycentric guidance maps a target valence–arousal (VA) point to weights over the three discrete emotion anchors of the enclosing triangle in a VA mesh. It then combines the editor's noise predictions for those emotion prompts with those weights. Optionally, a landmark-based 3-way face morph of the anchor edits guides the first few denoising steps.

Supported editors: **FLUX.2 [dev]**, **FLUX.2 [klein] 9B / 4B** and **Qwen-Image-Edit-2511**.

## Layout

| File | Contents |
| --- | --- |
| `src/main.py` | CLI and experiments (`--fid`, `--smoothness2d`) |
| `src/guidance.py` | `edit_va`: barycentric weights → (anchors → morph) → barycentric guidance or prompt interpolation |
| `src/pipelines/flux2.py` | Custom FLUX.2 dev/klein denoising loop and `Flux2Editor` wrapper |
| `src/pipelines/qwen.py` | Custom Qwen-Image-Edit-2511 denoising loop and `QwenEditor` wrapper |
| `src/face_morph.py` | Landmark detection and Delaunay-mesh 3-way face morphing |
| `src/dataset.py` | AffectNet and image-folder datasets |
| `src/utils.py` | VA mesh, barycentric interpolation, seeding |
| `src/va_mesh.csv` | Default VA mesh: emotion anchors and their (valence, arousal) positions |
| `notebooks/barycentric_guidance_klein.ipynb` | Colab demo: 5×5 VA grid for `notebooks/celeba_example.jpg` with FLUX.2 [klein] 4B |

## Installation

```bash
pip install -r requirements.txt
```

FLUX.2 [dev] is a gated model on Hugging Face: accept its license on the model page and run `hf auth login` before the first run. Weights are downloaded to `--cache_dir`.

## Usage

The default setup is FLUX.2 [dev] with barycentric guidance and face-morph refinement. This command generates a 10×10 VA grid (valence and arousal in [-1, 1]) for each of the first 100 neutral AffectNet validation images:

```bash
python src/main.py \
    --model flux2-dev \
    --mode bary_guide --morph --scale_w_cfg \
    --smoothness2d --total_images 100 \
    --guidance_scale 4.0 --morph_step 10 --anchor_steps 30 --inference_steps 30 \
    --img_dir /path/to/affectnet/Manually_Annotated_Images \
    --affectnet_csv /path/to/affectnet/validation.csv \
    --output_dir /path/to/outputs/flux2_dev_bary_guide \
    --cache_dir /path/to/hf_cache
```

The guidance and step values shown are the defaults, so the command still works without them. Images are saved as `<output_dir>/smoothness2d/<image>_<valence>_<arousal>.jpg`. Re-running the same command skips images that already exist, so an interrupted job can be resumed.

### Experiments

| Flag | Output | Images |
| --- | --- | --- |
| `--smoothness2d` | `<output_dir>/smoothness2d/` | 10×10 VA grid per source image (neutral faces only on AffectNet) |
| `--fid` | `<output_dir>/fid/` | One image per source at a random VA target in the unit disc, for FID |

`--total_images` sets how many source images are used. Both flags can be passed together.

```bash
# FID images with the same method
python src/main.py --model flux2-dev --mode bary_guide --morph --scale_w_cfg --fid --total_images 1000 \
    --img_dir /path/to/affectnet/Manually_Annotated_Images --affectnet_csv /path/to/affectnet/validation.csv \
    --output_dir /path/to/outputs/flux2_dev_bary_guide --cache_dir /path/to/hf_cache
```

### Methods

| `--mode` | Method |
| --- | --- |
| `bary_guide` (default) | Barycentric guidance: weighted sum of the three emotion prompts' noise predictions |
| `prompt_interp` | Prompt interpolation baseline: weighted average of the three prompt embeddings |

`--morph` adds face-morph refinement to either mode: the three anchor emotion edits are morphed together and guide the first `--morph_step` denoising steps. Without `--morph`, `prompt_interp` uses the model's stock pipeline call. `--scale_w_cfg` scales the guidance scale with the distance of the VA target from neutral, from 1 at (0, 0) up to `--guidance_scale` at radius 1.

```bash
# Prompt interpolation baseline (no morph)
python src/main.py --model flux2-dev --mode prompt_interp --smoothness2d \
    --img_dir /path/to/affectnet/Manually_Annotated_Images --affectnet_csv /path/to/affectnet/validation.csv \
    --output_dir /path/to/outputs/flux2_dev_prompt_interp --cache_dir /path/to/hf_cache
```

### Other models and datasets

`--model` selects the editor: `flux2-dev` (default), `flux2-klein` (FLUX.2 [klein] base 9B), `flux2-klein-4b` (FLUX.2 [klein] base 4B, the smallest; ~17 GB) or `qwen-2511` (Qwen-Image-Edit-2511). Without `--affectnet_csv`, `--img_dir` is read as a plain folder of images, e.g. FFHQ or CelebA-HQ.

```bash
# Qwen-Image-Edit-2511 on a folder of face images
python src/main.py --model qwen-2511 --mode bary_guide --morph --scale_w_cfg --smoothness2d \
    --img_dir /path/to/celeba_hq --output_dir /path/to/outputs/qwen_bary_guide --cache_dir /path/to/hf_cache
```

### Custom VA mesh

The emotion anchors are read from `src/va_mesh.csv` (columns `Emotion,Pleasure_Mean,Arousal_Mean`). Pass `--mesh_csv my_mesh.csv` to use a different set of emotions or positions. Each emotion word is placed into the prompt *"Change this persons facial expression to be {emotion} while maintaining the same facial features, composition, lighting, and camera angle"*.

Run `python src/main.py --help` for all options.
