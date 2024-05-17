from torch import nn
import numpy as np
from diffusion import DiffusionModel
from ddpm import DDPMSampler
from tqdm import tqdm
import torch


HEIGH = 512
WIDTH = 512
LATENT_HEIGHT = HEIGH // 8
LATENT_WIDTHT = WIDTH // 8



def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    ):

    with torch.no_grad():
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        generator = torch.Generator()
        if seed:
            generator.manual_seed(seed)
        else:
            generator.seed()
        clip = models['clip']
        clip.to(device)
        if do_cfg:
            prompt_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
            prompt_emb = clip(prompt_tokens)
            uncond_prompt_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_prompt_tokens = torch.tensor(uncond_prompt_tokens, dtype=torch.long).to(device)
            uncond_prompt_emb = clip(uncond_prompt_tokens)
            prompt_emb = torch.cat([prompt_emb, uncond_prompt_emb], axis=0)
        else:
            prompt_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
            prompt_emb = clip(prompt_tokens)
        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError('Unkown sampler name.')
        to_idle(clip)
        latent_size = (1, 4, LATENT_HEIGHT, LATENT_WIDTHT)
        if input_image:
            encoder = models['encoder']
            encoder = encoder.to(device)
            image = input_image.resize((WIDTH, HEIGH))
            image_tensor = np.array(image)
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32, device=device)
            image = rescale(image_tensor, (0, 255), (-1, 1)).unsqueeze(0)
            image_tensor = image.permute(0, 3, 1, 2)
            # (bs, 3, H, W)
            input_noize = torch.randn(latent_size, generator=generator, device=device)
            model_out = encoder(image_tensor, input_noize)
            #(bs, 4, H_lattent, W_lattent)
            sampler.set_strength(strength)
            latent = sampler.add_noise(model_out, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latent = torch.randn(latent_size)
        diffusion = models['diffusion']
        diffusion = diffusion.to(device)
        timestamps = tqdm(sampler.timesteps)
        for i, ts in enumerate(timestamps):
            time_emb = get_time_embeddings(ts).to(device) # (1, 160 * 2)
            model_input = latent
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            out_diffusion = diffusion(model_input, prompt_emb, time_emb)
            if do_cfg:
                cond, uncond  = torch.chunk(out_diffusion, 2)
                out_diffusion = cfg_scale * (cond - uncond) + uncond 
            latent = sampler.step(ts, latent, out_diffusion)
        to_idle(diffusion)
        decoder = models['decoder']
        decoder = decoder.to(device)
        decode_image = decoder(latent)
        to_idle(decoder)
        rescaled_image = decode_image.permute(0, 2, 3, 1)
        image = rescale(rescaled_image, (-1, 1), (0, 255), clamp=True)
        image = image.to("cpu", torch.uint8).numpy()
        return image[0]




def get_time_embeddings(timestamp: int) -> torch.Tensor: 
    freqs = torch.pow(10_000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestamp], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(min=new_min, max=new_max)
    return x
        
        
        
            
            
            