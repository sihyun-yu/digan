import click
import sys; sys.path.extend(['.', 'src'])
from tqdm import tqdm
import random

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import torch
from einops import rearrange

import dnnlib
import legacy
from fvd.fvd import get_fvd_logits, frechet_distance
from fvd.download import load_i3d_pretrained
from training.networks import Generator
from training.dataset import ImageFolderDataset, UCF101Wrapper


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]

    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)

    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)

    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    return mmd

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--data_path', help='Path to the dataset', required=True)
@click.option('--n_trials', type=int, help='Number of evaluation', default=10, show_default=True)
@click.option('--num_videos', type=int, help='Number of images to evaluate', default=2048, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
def main(
    ctx: click.Context,
    network_pkl: str,
    data_path: str,
    n_trials: int,
    num_videos: int,
    seed: int,
    ):
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    assert 256 % ngpus == 0, f"Must have 256 % n_gpus == 0"

    torch.set_grad_enabled(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #################### Load model ########################################
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
        G.forward = Generator.forward.__get__(G, Generator)

    if 'UCF' in network_pkl or 'ucf' in network_pkl:
        dataset = UCF101Wrapper(data_path, False, 128, data_path, xflip=False, return_vid=True)
    else:
        dataset = ImageFolderDataset(data_path, 128, 16, train=False, return_vid=True, xflip=False)

    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(device)

    #################### Compute FVD ###############################
    fvds = []
    kvds = []
    pbar = tqdm(total=n_trials)

    for _ in range(n_trials):
        test_iterator = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True))
        fvd, kvd = eval_fvd(i3d, G, test_iterator, num_videos, device)
        fvds.append(fvd)
        kvds.append(kvd)

        pbar.update(1)
        fvd_mean = np.mean(fvds)
        kvd_mean = np.mean(kvds)
        fvd_std = np.std(fvds)
        kvd_std = np.std(kvds)

        pbar.set_description(f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, KVD {kvd_mean:.2f} +/- {kvd_std:.2f}")

    pbar.close()
    print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}")
    print(f"Final KVD {kvd_mean:.2f} +/- {kvd_std:.2f}")


def eval_fvd(i3d, G, iterator, num_videos, device):
    grid_z = torch.randn([num_videos, G.z_dim], device=device).split(1)

    fake_embeddings = []
    fake = torch.cat([rearrange(
                        G(z, None, timesteps=16, noise_mode='const')[0].clamp(-1, 1).cpu(),
                        '(b t) c h w -> b t h w c', t=16) for z in grid_z])

    fake = ((fake + 1.) / 2. * 255).type(torch.uint8)
    fake = fake.numpy()

    for i in range(num_videos):
        fake_embeddings.append(get_fvd_logits(fake[i:i+1], i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real_embeddings = []
    for i in range(num_videos):
        real = next(iterator)
        real = real.permute(0, 2, 3, 4, 1).cpu().numpy().astype('uint8') # BCTHW -> BTHWC
        real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))

    real_embeddings = torch.cat(real_embeddings)

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    kvd = polynomial_mmd(fake_embeddings.clone().detach().cpu().numpy(), real_embeddings.detach().cpu().numpy())
    return fvd.item(), kvd.item()


if __name__ == '__main__':
    main()
