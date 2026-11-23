"""Qwen-Image-Edit-2511 editor with barycentric + face-morph guidance."""
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


def _rescale(pred, ref):
    """Rescale pred to have the per-token norm of ref (as in Qwen's true CFG)."""
    return pred * (torch.norm(ref, dim=-1, keepdim=True) / torch.norm(pred, dim=-1, keepdim=True))


@torch.no_grad()
def generate(
    self,
    image,
    morphed_image=None,
    prompt_dicts=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    negative_prompt=None,
    true_cfg_scale=4.0,
    num_inference_steps=50,
    num_morph_guidance_steps=18,
    generator=None,
    max_sequence_length=512,
):
    """Altered QwenImageEditPlus __call__ with barycentric and face-morph guidance.

    - For the first `num_morph_guidance_steps` steps (if `morphed_image` is given) the noise
      prediction is replaced by (latents - morphed_latents) / sigma.
    - prompt_dicts {prompt: weight}: barycentric guidance, noise_pred = sum_i w_i * D(x_t, y_i),
      rescaled to the norm of the first term.
    - Otherwise `prompt_embeds`/`prompt_embeds_mask` are used, with true CFG against
      `negative_prompt` when true_cfg_scale > 1.
    """
    device = self._execution_device
    self._guidance_scale = None
    self._attention_kwargs = {}
    self._current_timestep = None
    self._interrupt = False

    # 1. Preprocess source image (VLM condition image + VAE image) and morph target
    img_w, img_h = image.size
    cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
    vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, img_w / img_h)
    multiple_of = self.vae_scale_factor * 2
    width, height = vae_w // multiple_of * multiple_of, vae_h // multiple_of * multiple_of
    condition_images = [self.image_processor.resize(image, cond_h, cond_w)]
    vae_image = self.image_processor.preprocess(image, vae_h, vae_w).unsqueeze(2)

    if morphed_image is not None:
        m_w, m_h = calculate_dimensions(VAE_IMAGE_SIZE, morphed_image.size[0] / morphed_image.size[1])
        morphed_vae_image = self.image_processor.preprocess(morphed_image, m_h, m_w).unsqueeze(2)

    # 2. Text embeddings (Qwen's VLM encoder is conditioned on the source image)
    def encode(prompt):
        return self.encode_prompt(image=condition_images, prompt=prompt, device=device,
                                  max_sequence_length=max_sequence_length)

    do_true_cfg = False
    if prompt_dicts is not None:
        prompts = list(prompt_dicts.keys())
        weights = torch.tensor(list(prompt_dicts.values()), device=device).bfloat16()
        cond = [encode(p) for p in prompts]
        prompt_embeds = cond[0][0]
    else:
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None  # matches encode_prompt's handling of an all-ones mask
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        if do_true_cfg:
            neg_embeds, neg_mask = encode(negative_prompt)

    # 3. Latents
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        [vae_image], 1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, None,
    )
    if morphed_image is not None:
        _, morphed_latents = self.prepare_latents(
            [morphed_vae_image], 1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, None,
        )
    img_shapes = [[
        (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
        (1, vae_h // self.vae_scale_factor // 2, vae_w // self.vae_scale_factor // 2),
    ]]

    # 4. Timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        latents.shape[1],
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    self._num_timesteps = len(timesteps)

    # 5. Denoising loop
    self.scheduler.set_begin_index(0)
    for i, t in enumerate(self.progress_bar(timesteps)):
        self._current_timestep = t
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_model_input = torch.cat([latents, image_latents], dim=1)

        def predict(embeds, mask):
            out = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,  # Qwen-Image-Edit is not guidance-distilled
                encoder_hidden_states_mask=mask,
                encoder_hidden_states=embeds,
                img_shapes=img_shapes,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            return out[:, :latents.size(1)]

        if morphed_image is not None and i < num_morph_guidance_steps:
            # Face-morph guidance: drive latents toward the morphed image
            if self.scheduler.step_index is None:
                self.scheduler._init_step_index(timestep)
            sigma = self.scheduler.sigmas[self.scheduler.step_index]
            noise_pred = (latents - morphed_latents) / sigma
        elif prompt_dicts is not None:
            # Barycentric guidance: weighted sum over prompt predictions
            preds = [predict(e, m) for e, m in cond]
            noise_pred = sum(w * p for w, p in zip(weights, preds))
            noise_pred = _rescale(noise_pred, preds[0])
        else:
            noise_pred = predict(prompt_embeds, prompt_embeds_mask)
            if do_true_cfg:
                neg_pred = predict(neg_embeds, neg_mask)
                noise_pred = _rescale(neg_pred + true_cfg_scale * (noise_pred - neg_pred), noise_pred)

        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    self._current_timestep = None

    # 6. Decode
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor).to(self.vae.dtype)
    z_dim = self.vae.config.z_dim
    latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = self.image_processor.postprocess(image, output_type='pil')

    self.maybe_free_model_hooks()
    return QwenImagePipelineOutput(images=image)


class QwenEditor:
    """Wraps Qwen-Image-Edit-2511 with the operations barycentric guidance needs."""

    negative_prompt = " "

    def __init__(self, cache_dir, device):
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        self.pipe.generate = generate.__get__(self.pipe)
        self.pipe.to(device)
        self.device = device

    def _generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    def edit(self, image, prompt, guidance_scale, steps, seed):
        """Plain single-prompt edit (used for the anchor emotion images)."""
        return self.pipe(
            image=[image], prompt=prompt, generator=self._generator(seed),
            true_cfg_scale=guidance_scale, negative_prompt=self.negative_prompt,
            num_inference_steps=steps, guidance_scale=1.0, num_images_per_prompt=1,
        ).images[0]

    def bary_guide(self, image, morphed, prompt_dict, steps, morph_steps, seed):
        """Barycentric guidance: noise_pred = sum_i w_i * D(x_t, y_i) over prompt_dict {prompt: w_i}."""
        return self.pipe.generate(
            image=image, morphed_image=morphed, prompt_dicts=prompt_dict,
            num_inference_steps=steps, num_morph_guidance_steps=morph_steps, generator=self._generator(seed),
        ).images[0]

    @torch.no_grad()
    def _interpolated_embeds(self, image, prompts, weights):
        """Encode each prompt with the source image, zero-pad to equal length, interpolate."""
        img_w, img_h = image.size
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
        cond_img = self.pipe.image_processor.resize(image, cond_h, cond_w)

        embeds, masks = [], []
        for p in prompts:
            emb, msk = self.pipe.encode_prompt(image=[cond_img], prompt=p, device=self.device)
            embeds.append(emb[0].cpu().float())
            masks.append((msk if msk is not None else torch.ones(1, emb.shape[1]))[0].cpu().float())

        max_len = max(e.shape[0] for e in embeds)
        padded_embeds = torch.zeros(len(prompts), max_len, embeds[0].shape[-1])
        padded_masks = torch.zeros(len(prompts), max_len)
        for k, (emb, msk) in enumerate(zip(embeds, masks)):
            padded_embeds[k, :emb.shape[0]] = emb
            padded_masks[k, :msk.shape[0]] = msk

        w = torch.tensor(weights, dtype=torch.float32)
        interp_emb = (w[:, None, None] * padded_embeds).sum(0, keepdim=True).to(self.device).bfloat16()
        # Union mask: attend to any position valid in at least one anchor
        interp_mask = padded_masks.max(dim=0).values.unsqueeze(0).to(self.device).int()
        return interp_emb, interp_mask

    def prompt_interp(self, image, morphed, prompts, weights, guidance_scale, steps, morph_steps, seed):
        """Prompt-embedding interpolation with barycentric weights (stock __call__ if no morph image)."""
        interp_emb, interp_mask = self._interpolated_embeds(image, prompts, weights)
        kwargs = dict(
            prompt_embeds=interp_emb, prompt_embeds_mask=interp_mask, true_cfg_scale=guidance_scale,
            negative_prompt=self.negative_prompt, num_inference_steps=steps, generator=self._generator(seed),
        )
        if morphed is None:
            return self.pipe(image=[image], guidance_scale=1.0, num_images_per_prompt=1, **kwargs).images[0]
        return self.pipe.generate(image=image, morphed_image=morphed, num_morph_guidance_steps=morph_steps, **kwargs).images[0]
