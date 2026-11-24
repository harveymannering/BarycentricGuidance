"""VA editing by barycentric guidance or prompt interpolation, with optional face-morph refinement."""
import numpy as np
import torch
from PIL import Image

from face_morph import detect_landmarks, morph_face_tri
from utils import get_triangular_interpolation


MODES = ['bary_guide', 'prompt_interp']


def edit_va(editor, source_image, va_target, points, tri, emotion_prompts, args, seed=42, emotion_cache=None):
    """Edit `source_image` toward the VA point `va_target`.

    1. Find the enclosing mesh triangle and its barycentric weights.
    2. (--morph) Generate the 3 anchor emotion edits (cached in `emotion_cache` across
       calls for the same source), detect landmarks and build a 3-way face morph.
    3. Edit with `args.mode`, using the morph (if any) for the first `morph_step` steps:
       - 'bary_guide':    barycentric guidance, a w-weighted sum of the 3 emotion prompts'
                          noise predictions (+ CFG via guidance_scale).
       - 'prompt_interp': w-weighted interpolation of the 3 prompt embeddings.

    Returns (PIL image, emotion_cache) where emotion_cache maps vertex idx -> (image, landmarks).
    """
    emotion_cache = {} if emotion_cache is None else emotion_cache

    indices, bary = get_triangular_interpolation(va_target.reshape(1, 2), points, tri)
    idx, w = indices[0], bary[0]
    prompts = [emotion_prompts[j] for j in idx]

    guidance_scale = args.guidance_scale
    if args.scale_w_cfg:
        guidance_scale = 1 + (guidance_scale - 1) * np.linalg.norm(va_target)

    morphed = None
    if args.morph:
        try:
            for j in idx:
                if j not in emotion_cache:
                    emo_img = editor.edit(source_image, emotion_prompts[j], args.guidance_scale,
                                          args.anchor_steps, seed)
                    emotion_cache[j] = (emo_img, detect_landmarks(np.array(emo_img)))
                    torch.cuda.empty_cache()
            morphed = Image.fromarray(morph_face_tri(
                [np.array(emotion_cache[j][0]) for j in idx], [emotion_cache[j][1] for j in idx], w,
            ))
        except Exception as e:
            # e.g. face not found in an anchor image: fall back to no morph guidance
            print(f"Failed to detect one or more face: {e}", flush=True)

    if args.mode == 'bary_guide':
        prompt_dict = {p: wi * guidance_scale for p, wi in zip(prompts, w)}
        if guidance_scale != 1.0:
            prompt_dict[editor.negative_prompt] = 1 - guidance_scale
        img = editor.bary_guide(source_image, morphed, prompt_dict, args.inference_steps, args.morph_step, seed)
    else:
        img = editor.prompt_interp(source_image, morphed, prompts, w, guidance_scale, args.inference_steps,
                                   args.morph_step, seed)
    torch.cuda.empty_cache()
    return img, emotion_cache
