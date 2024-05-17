from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import DiffusionModel
from model_converter import load_from_standard_weights


def load_pretaining_weights(ckpt_path, device):
    state_dict = load_from_standard_weights(ckpt_path, device)
    
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'])

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'])

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'])

    diffusion = DiffusionModel().to(device)
    diffusion.load_state_dict(state_dict['diffusion'])
    return {
        'clip': clip,
        'decoder': decoder,
        'encoder': encoder,
        'clip': clip,
        'diffusion': diffusion 
    }