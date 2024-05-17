import torch
import numpy as np
import PIL
import transformers
from transformers import CLIPTokenizer
from model_loader import load_pretaining_weights
import pipeline
from PIL import Image


class Config:
    def __init__(self):
        self.tokenizer = CLIPTokenizer(r"C:\Users\zinov\StableDiff_scratch\data\tokenizer_vocab.json", merges_file=r"C:\Users\zinov\StableDiff_scratch\data\tokenizer_merges.txt")
        self.models = {}
        self.prompt = "man, student, nvinkpunk"
        self.uncond_prompt=''
        self.do_cfg = True
        self.cfg_scale = 7  # min: 1, max: 14
        self.strength = 0.4
        self.sampler = "ddpm"
        self.num_inference_steps = 60
        self.seed = 42
        self.DEVICE = 'cpu'


def opp_malevich(prompt, 
                 cfg,
                 path2image=None,
                uncond_prompt=''):
    print(path2image, 'opp')
    if path2image:
        input_image = Image.open(path2image)
    else:
        input_image = None
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=cfg.strength,
        do_cfg=cfg.do_cfg,
        cfg_scale=cfg.cfg_scale,
        sampler_name=cfg.sampler,
        n_inference_steps=cfg.num_inference_steps,
        seed=cfg.seed,
        models=cfg.models,
        device=cfg.DEVICE,    idle_device="cpu",
        tokenizer=cfg.tokenizer,
    )
    print('diff ok')
    return Image.fromarray(output_image)
    
    
    