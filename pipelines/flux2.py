"""FLUX.2 [dev] and FLUX.2 [klein] editors with barycentric + face-morph guidance."""
import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

MODEL_IDS = {
    'flux2-dev': "black-forest-labs/FLUX.2-dev",
    'flux2-klein': "black-forest-labs/FLUX.2-klein-base-9B",
    'flux2-klein-4b': "black-forest-labs/FLUX.2-klein-base-4B",
}


def _preprocess(self, img):
    """Resize to <= 1MP, snap to a multiple of the latent patch size, convert to tensor."""
    width, height = img.size
    if width * height > 1024 * 1024:
        img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
        width, height = img.size
    multiple_of = self.vae_scale_factor * 2
    width, height = (width // multiple_of) * multiple_of, (height // multiple_of) * multiple_of
    return self.image_processor.preprocess(img, height=height, width=width, resize_mode='crop'), height, width


@torch.no_grad()
def generate(
    self,
    image,
    morphed_image=None,
    prompt_dicts=None,
    prompt_embeds=None,
    negative_prompt=None,
    guidance_scale=4.0,
    num_inference_steps=14,
    num_morph_guidance_steps=5,
    generator=None,
    max_sequence_length=512,
):
    """Altered FLUX.2 __call__ (dev and klein) with barycentric and face-morph guidance.

    - For the first `num_morph_guidance_steps` steps (if `morphed_image` is given) the noise
      prediction is replaced by (latents - morphed_latents) / sigma, driving the latents
      toward the morphed image.
    - prompt_dicts {prompt: weight}: barycentric guidance, noise_pred = sum_i w_i * D(x_t, y_i).
    - Otherwise `prompt_embeds` (e.g. interpolated) is used, with standard CFG against
      `negative_prompt` if one is given.
    """
    is_klein = isinstance(self, Flux2KleinPipeline)
    device = self._execution_device
    self._guidance_scale = guidance_scale
    self._attention_kwargs = None
    self._current_timestep = None
    self._interrupt = False

    # 1. Text embeddings
    if prompt_dicts is not None:
        prompts = list(prompt_dicts.keys())
        weights = torch.tensor(list(prompt_dicts.values()), device=device).bfloat16()
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompts, device=device, max_sequence_length=max_sequence_length,
        )
    else:
        # Pre-computed (interpolated) embeddings
        text_ids = torch.zeros(1, prompt_embeds.shape[1], 4, device=device, dtype=prompt_embeds.dtype)
        neg_embeds = neg_text_ids = None
        if negative_prompt is not None:
            neg_embeds, neg_text_ids = self.encode_prompt(
                prompt=negative_prompt, device=device, max_sequence_length=max_sequence_length,
            )

    # 2. Latents: noise, source-image conditioning and (optional) morph target
    cond_image, height, width = _preprocess(self, image)
    latents, latent_ids = self.prepare_latents(
        batch_size=1,
        num_latents_channels=self.transformer.config.in_channels // 4,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    image_latents, image_latent_ids = self.prepare_image_latents(
        images=[cond_image], batch_size=1, generator=generator, device=device, dtype=self.vae.dtype,
    )
    morphed_latents = None
    if morphed_image is not None:
        morphed_latents, _ = self.prepare_image_latents(
            images=[_preprocess(self, morphed_image)[0]], batch_size=1,
            generator=generator, device=device, dtype=self.vae.dtype,
        )

    # 3. Timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(self.scheduler.config, 'use_flow_sigmas', False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    self._num_timesteps = len(timesteps)

    # dev embeds a distilled guidance scale (fixed, separate from CFG); klein has none
    guidance = None if is_klein else torch.full([latents.shape[0]], 4.0, device=device, dtype=torch.float32)

    # 4. Denoising loop
    self.scheduler.set_begin_index(0)
    for i, t in enumerate(self.progress_bar(timesteps)):
        self._current_timestep = t
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        def predict(embeds, txt_ids):
            out = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=embeds,
                txt_ids=txt_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            return out[:, :latents.size(1)]

        if morphed_latents is not None and i < num_morph_guidance_steps:
            # Face-morph guidance: drive latents toward the morphed image
            if self.scheduler.step_index is None:
                self.scheduler._init_step_index(timestep)
            sigma = self.scheduler.sigmas[self.scheduler.step_index]
            noise_pred = (latents - morphed_latents) / sigma
        elif prompt_dicts is not None:
            # Barycentric guidance: weighted sum over prompt predictions
            noise_pred = 0
            for j in range(len(prompts)):
                noise_pred = noise_pred + weights[j] * predict(prompt_embeds[j:j + 1], text_ids[j:j + 1])
        else:
            noise_pred = predict(prompt_embeds, text_ids)
            if neg_embeds is not None:
                noise_pred_neg = predict(neg_embeds, neg_text_ids)
                noise_pred = noise_pred_neg + guidance_scale * (noise_pred - noise_pred_neg)

        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    self._current_timestep = None

    # 5. Decode (unpack with ids, BN de-normalise, unpatchify)
    latents = self._unpack_latents_with_ids(latents, latent_ids)
    bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = self._unpatchify_latents(latents * bn_std + bn_mean)
    image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type='pil')

    self.maybe_free_model_hooks()
    return Flux2PipelineOutput(images=image)


class Flux2Editor:
    """Wraps a FLUX.2 dev / klein pipeline with the operations barycentric guidance needs."""

    negative_prompt = ""

    def __init__(self, model, cache_dir, device):
        self.is_klein = model.startswith('flux2-klein')
        pipeline_cls = Flux2KleinPipeline if self.is_klein else Flux2Pipeline
        self.pipe = pipeline_cls.from_pretrained(MODEL_IDS[model], torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        self.pipe.generate = generate.__get__(self.pipe)
        self.pipe.to(device)
        self.device = device
        self._embeddings = {}

    def _generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def _embed(self, prompt):
        # FLUX.2 text embeddings don't depend on the source image, so cache them per prompt
        if prompt not in self._embeddings:
            self._embeddings[prompt] = self.pipe.encode_prompt(prompt)[0].cpu().float()
        return self._embeddings[prompt]

    def edit(self, image, prompt, guidance_scale, steps, seed):
        """Plain single-prompt edit (used for the anchor emotion images)."""
        return self.pipe(
            image=image, prompt=prompt, guidance_scale=guidance_scale,
            num_inference_steps=steps, generator=self._generator(seed),
        ).images[0]

    def bary_guide(self, image, morphed, prompt_dict, steps, morph_steps, seed):
        """Barycentric guidance: noise_pred = sum_i w_i * D(x_t, y_i) over prompt_dict {prompt: w_i}."""
        return self.pipe.generate(
            image=image, morphed_image=morphed, prompt_dicts=prompt_dict,
            num_inference_steps=steps, num_morph_guidance_steps=morph_steps, generator=self._generator(seed),
        ).images[0]

    def prompt_interp(self, image, morphed, prompts, weights, guidance_scale, steps, morph_steps, seed):
        """Prompt-embedding interpolation with barycentric weights (stock __call__ if no morph image)."""
        embeds = torch.cat([self._embed(p) for p in prompts])
        interp = (torch.tensor(weights, dtype=torch.float32)[:, None, None] * embeds).sum(0, keepdim=True)
        interp = interp.to(self.device).bfloat16()
        if morphed is None:
            # Model's default __call__ (dev: guidance_scale is the distilled guidance embedding)
            return self.pipe(
                image=image, prompt_embeds=interp, guidance_scale=guidance_scale,
                num_inference_steps=steps, generator=self._generator(seed),
            ).images[0]
        # klein (undistilled) does true CFG against "" when guidance_scale > 1, as in its __call__
        negative = self.negative_prompt if self.is_klein and guidance_scale > 1.0 else None
        return self.pipe.generate(
            image=image, morphed_image=morphed, prompt_embeds=interp, negative_prompt=negative,
            guidance_scale=guidance_scale, num_inference_steps=steps, num_morph_guidance_steps=morph_steps,
            generator=self._generator(seed),
        ).images[0]
