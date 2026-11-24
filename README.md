# Barycentric Guidance

[![arXiv](https://img.shields.io/badge/arXiv-2610.16892-b31b1b.svg)](https://arxiv.org/abs/2610.16892) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harveymannering/BarycentricGuidance/blob/main/notebooks/barycentric_guidance_klein.ipynb)

Code for *Barycentric Guidance: Turning Foundation Image Editors into Continuous Affective Controllers* (NeurIPS 2026).

Barycentric guidance enables valence-arousal (VA) based image editing by mapping combinations of discrete emotion prompts to VA points. This repo focuses on the core experiment of the paper, facial expression editing with FLUX.2-dev, and includes both the baseline Barycentric Guidance method and latent-space face morphing to improve smoothness. Other supported editors include **FLUX.2-klein-4B**, **FLUX.2-klein-9B**, and **Qwen-Image-Edit-2511**.

## Installation

```bash
pip install -r requirements.txt
```

FLUX.2-dev is a gated model on Hugging Face, so accept its license on the model page and run `hf auth login` before the first run. For a lightweight version of our method, you can try running our code with FLUX.2-klein-4B, which does not require accepting a license. However, results with this model will be weaker than those reported in the paper. Weights should be downloaded to `--cache_dir`.

## Usage

The default setup is FLUX.2-dev with barycentric guidance and face morphing. This command generates a 10×10 VA grid (valence and arousal in [-1, 1]) for each of the first 100 images in a folder of CelebA-HQ faces. CelebA-HQ images can be downloaded from the [CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

```bash
python src/main.py \
    --model flux2-dev \
    --mode bary_guide --morph --scale_w_cfg \
    --smoothness2d --total_images 100 \
    --img_dir /path/to/celeba_hq \
    --output_dir /path/to/outputs/flux2_dev_bary_guide \
    --cache_dir /path/to/hf_cache
```

Images are saved as `<output_dir>/smoothness2d/<image>_<valence>_<arousal>.jpg`. Re-running the same command skips images that already exist, so an interrupted job can be resumed. Run `python src/main.py --help` for all options.

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | `flux2-dev` | Editor: `flux2-dev`, `flux2-klein` (9B), `flux2-klein-4b` or `qwen-2511` (Qwen-Image-Edit-2511) |
| `--img_dir` | required | Folder of source face images (e.g. CelebA-HQ or FFHQ) |
| `--affectnet_csv` | none | AffectNet annotation CSV; if given, `--img_dir` is read as AffectNet and filtered to neutral faces |
| `--output_dir` | required | Where generated images are saved |
| `--cache_dir` | required | Where model weights are downloaded |
| `--mesh_csv` | `src/va_mesh.csv` | VA mesh of emotion anchors (columns `Emotion,Pleasure_Mean,Arousal_Mean`) |
| `--smoothness2d` | off | Generate a 10×10 VA grid per source image, saved to `<output_dir>/smoothness2d/` |
| `--fid` | off | Generate one image per source at a random VA target in the unit disc, saved to `<output_dir>/fid/` |
| `--total_images` | `100` | Number of source images used by `--smoothness2d` / `--fid` |
| `--mode` | `bary_guide` | `bary_guide`: barycentric guidance (ours); `prompt_interp`: prompt-embedding interpolation baseline |
| `--morph` | off | Morph the three anchor emotion edits together and use the result to guide the first `--morph_step` denoising steps |
| `--scale_w_cfg` | off | Scale the guidance scale from 1 at neutral (0, 0) up to `--guidance_scale` at radius 1 |
| `--guidance_scale` | `4.0` | Guidance scale for the anchor and final edits |
| `--morph_step` | `10` | Number of early denoising steps guided by the morphed image |
| `--anchor_steps` | `30` | Denoising steps for the anchor emotion images |
| `--inference_steps` | `30` | Denoising steps for the final images |

## Notebooks

We also include two short Jupyter notebooks:
- `notebooks/barycentric_guidance_klein.ipynb` is a self-contained Colab demo that edits a CelebA-HQ image over a 5×5 VA grid using the lightweight FLUX.2-klein-4B model. 
- `notebooks/clip_pca.ipynb` runs PCA on CLIP text embeddings of emotion words and shows that the resulting low-dimensional structure closely matches the valence-arousal model.

## Citation

```bibtex
@inproceedings{mannering2026barycentric,
  title     = {Barycentric Guidance: Turning Foundation Image Editors
               into Continuous Affective Controllers},
  author    = {Mannering, Harvey and Huang, Zhiwu and Pr{\"u}gel-Bennett, Adam},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
