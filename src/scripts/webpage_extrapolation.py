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
from scripts import save_image_grid
from scripts.custom_network.network_super_resol import Generator
from einops import rearrange

import torch.nn.functional as F
import numpy as np
torch.set_grad_enabled(False)

#----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--seed', type=int, help='Random seed', default=0, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(
    ctx: click.Context,
    seed: int,
    outdir: str,
):
    mocogan = np.load('/home/jihoon/workspace/inr-gan/mocogan_extrapolation.npz')
    ours = np.load('/home/jihoon/workspace/inr-gan/ours_extrapolation.npz')

    mocogan = torch.from_numpy(mocogan)
    ours = torch.from_numpy(ours)

    total = torch.cat([mocogan, ours], dim=-1)
    total = rearrange(total, 'c t h w -> t h w c')

    total[16:, 0:2, :, 0:1] = 192
    total[16:, 126:128, :, 0:1] = 192
    total[16:, :, 0:2, 0:1] = 192
    total[16:, :, 128+126:128+128, 0:1] = 192


    total[16:, 0:2, :, 1:] = 0
    total[16:, 126:128:, :, 1:] = 0
    total[16:, :, 0:2, 1:] = 0
    total[16:, :, 128+126:128+128, 1:] = 0


    torchvision.io.write_video(os.path.join(outdir, f'webpage_extrapolation.mp4'), total, fps=8)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
