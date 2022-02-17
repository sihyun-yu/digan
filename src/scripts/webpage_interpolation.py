"""Generates a dataset of images using pretrained network pickle."""
import math
import sys; sys.path.extend(['.', 'src'])
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torchvision

import legacy
import torch_utils.misc as misc
from hydra.experimental import compose, initialize
#from scripts import save_image_grid
from scripts.custom_network.network_super_resol import Generator
from einops import rearrange

torch.set_grad_enabled(False)


#----------------------------------------------------------------------------

def load_generator(img_resol, super_resolution):
    initialize(config_path="../../configs", job_name="INR-GAN training")
    hydra_cfg = compose(config_name='inr-gan-learnable.yml')
    synthesis_kwargs = dnnlib.EasyDict()
    synthesis_kwargs.channel_base = int(hydra_cfg.generator.get('fmaps', -1) * 32768)
    synthesis_kwargs.channel_max = 512
    synthesis_kwargs.num_fp16_res = 4
    synthesis_kwargs.conv_clamp = 256
    if hydra_cfg.generator.get('fp32'):
        synthesis_kwargs.num_fp16_res = 0
        synthesis_kwargs.conv_clamp = None

    mapping_kwargs = dnnlib.EasyDict()
    mapping_kwargs.num_layers = hydra_cfg.generator.get('mapping_net_n_layers', 2)
    cfg = OmegaConf.to_container(hydra_cfg.generator)
    G_super = Generator(z_dim=512, w_dim=512, ours=True, img_channels=3, img_resolution=img_resol,
                        c_dim=0, mapping_kwargs=mapping_kwargs, sr_ratio=super_resolution,
                        synthesis_kwargs=synthesis_kwargs, cfg=cfg)

    return G_super


def save_as_images(G, outdir, timesteps=32, ws=None):
    device = torch.device('cuda')
    z = torch.randn([1, G.z_dim], device=device)

    if ws is None:
        ws = G.mapping(z, None, truncation_psi=1, truncation_cutoff=None)
    Ts = torch.linspace(0, 1, steps=timesteps).view(timesteps, 1, 1)
    Ts = Ts.unsqueeze(0).repeat(1, 1, 1, 1).view(-1, 1, 1, 1).to(device)
    z_motion = torch.randn(1, 512).to(device)

    images = []
    for i in range(int(timesteps/16)):
        Ts_part = Ts[16 * i: 16 * (i+1)]
        img = rearrange(G.synthesis(ws, Ts_part, z_motion, noise_mode='const').detach().cpu(),
                        '(b t) c h w -> b t c h w', t=16)
        images.append(img)
    images = torch.cat(images, dim=1)
    images = (images + 1) * (255. / 2.)
    images = images.clamp(0, 255)
    images = rearrange(images, 'b t c h w -> b t h w c')[0]

    images_for_save = rearrange(images, 't h w c -> t c h w') / 255.
    torchvision.io.write_video(os.path.join(outdir, f'taichi128_{timesteps}.mp4'), images, fps=32)
    torchvision.utils.save_image(images_for_save, os.path.join(outdir, f'taichi128_{timesteps}.png'), nrow=16)

    return ws


def save_image_grid(img, fname, drange, grid_size, normalize=True):
    # img.shape == b c t h w, 16 * 256, 3, 128, 128

    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [1, 3]

    if C == 1:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=64)
        imgs = [PIL.Image.fromarray(img[i, :, 0], 'L') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=64)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def interpolate_forward(G, z, timesteps=16):
    device = torch.device('cuda')

    ws = G.mapping(z, None, truncation_psi=1, truncation_cutoff=None)
    Ts = torch.linspace(0, 1, steps=timesteps).view(timesteps, 1, 1)
    Ts = Ts.unsqueeze(0).repeat(1, 1, 1, 1).view(-1, 1, 1, 1).to(device)
    z_motion = torch.randn(1, 512).to(device)

    images = []
    for i in range(int(timesteps/16)):
        Ts_part = Ts[16 * i: 16 * (i+1)]
        img = rearrange(G.synthesis(ws, Ts_part, z_motion, noise_mode='const').detach().cpu(),
                        '(b t) c h w -> b t c h w', t=16)
        images.append(img)
    images = torch.cat(images, dim=1)
    images = (images + 1) * (255. / 2.)
    images = images.clamp(0, 255)
    images = rearrange(images, 'b t c h w -> b t h w c')[0]

    return images

#----------------------------------------------------------------------------

def generate_video(G, outdir, grid_size, grid_z, img_resolution=128, timesteps=16):
    # t h w c
    device = torch.device('cuda')
    idx = 2
    z = torch.randn([512, G.z_dim], device=device)
    z = z[idx:idx+1]


    images_16 = interpolate_forward(G, z, timesteps=16).unsqueeze(1).repeat(1,8,1,1,1).numpy()
    images_16 = rearrange(images_16, 't r h w c -> (t r) h w c')

    images_128 = interpolate_forward(G, z, timesteps=128).numpy()
    print(images_128.shape)

    images = np.concatenate([images_16, images_128], axis=-2)
    images = rearrange(np.expand_dims(images, axis=0), 'b t h w c -> b c t h w')

    #  b c t h w
    save_image_grid(images / 255., os.path.join(outdir, f'webpage_interpolation.gif'),
                    drange=[0, 1], grid_size=(1,1))

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--external_truncation_psi', type=float, help='External truncation psi (no w_avg)', default=1.0, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--timesteps', type=int, help='Timesteps', default=32, show_default=True)
@click.option('--super_resolution', type=int, help='Super resolution ratio', default=2, show_default=True)
@click.option('--num_videos', type=int, help='Number of videos to generate', default=16, show_default=True)
@click.option('--batch_size', type=int, help='Batch size to use for generation', default=4, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    external_truncation_psi: float,
    noise_mode: str,
    timesteps: int,
    super_resolution: int,
    num_videos: int,
    batch_size: int,
    seed: int,
    outdir: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
        G.forward = Generator.forward.__get__(G, Generator)
        print("Done. ")

    os.makedirs(outdir, exist_ok=True)
    img_resolution = G.img_resolution

    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # define grid size (for gif), e.g., (4, 4) will generate 16 videos in one gif file
    grid_size = (int(math.sqrt(num_videos)), int(math.sqrt(num_videos)))
    grid_z = torch.randn([grid_size[0] * grid_size[1], G.z_dim], device=device).split(1)

    # Save original video, i.e., 128 resolution video with 16 frames
    generate_video(G, outdir, grid_size, grid_z, img_resolution=img_resolution, timesteps=16)



if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
