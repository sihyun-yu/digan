import copy
import numpy as np

import torch
from einops import rearrange

import dnnlib
from scripts.fvd.pytorch_i3d import InceptionI3d
from scripts.fvd.fvd import get_fvd_logits, frechet_distance
from scripts.fvd.download import download

_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'


def _load_i3d_pretrained(device=torch.device('cpu')):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d


def _eval_fvd(opts, i3d, G, dataset, device, num_videos=512):
    num_videos_per_gpu = (num_videos - 1) // opts.num_gpus + 1
    grid_z = torch.randn([num_videos_per_gpu, G.z_dim], device=device).split(1)

    if G.c_dim > 0:
        grid_c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(num_videos_per_gpu)]
        grid_c = torch.from_numpy(np.stack(grid_c)).pin_memory().to(opts.device).split(1)
    else:
        grid_c = [None] * num_videos_per_gpu

    fake_embeddings = []
    fake = torch.cat([rearrange(
        G(z, c, timesteps=16, noise_mode='const')[0].clamp(-1, 1).cpu(),
        '(b t) c h w -> b t h w c', t=16) for z, c in zip(grid_z, grid_c)])

    fake = ((fake + 1.) / 2. * 255).type(torch.uint8)
    fake = fake.numpy()

    for i in range(num_videos_per_gpu):
        fake_embeddings.append(get_fvd_logits(fake[i:i+1], i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real_embeddings = []

    item_subset = [(i * opts.num_gpus + opts.rank) % num_videos for i in range((num_videos - 1) // opts.num_gpus + 1)]
    for real in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset,
                                            batch_size=1, pin_memory=False, num_workers=0):
        real = real.permute(0, 2, 3, 4, 1).cpu().numpy().astype('uint8')  # BCTHW -> BTHWC
        real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))

    real_embeddings = torch.cat(real_embeddings)

    if opts.num_gpus > 1:  # if distributed, gather computed tensor to each gpus
        gather_fake = [torch.zeros_like(fake_embeddings) for _ in range(opts.num_gpus)]
        torch.distributed.all_gather(gather_fake, fake_embeddings)
        fake_embeddings = torch.cat(gather_fake)

        gather_real = [torch.zeros_like(real_embeddings) for _ in range(opts.num_gpus)]
        torch.distributed.all_gather(gather_real, real_embeddings)
        real_embeddings = torch.cat(gather_real)

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    return fvd.item()


def compute_fvd(opts, num_videos=512, num_iter=5):
    # load vid classifier
    i3d = _load_i3d_pretrained(opts.device)

    # load dataloader
    dataset = dnnlib.util.construct_class_by_name(return_vid=True,
                                                  **opts.dataset_kwargs)  # return_vid True for FVD measure

    # Copy generator
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # compute fvd
    fvd = [_eval_fvd(opts, i3d, G, dataset, opts.device, num_videos=num_videos) for _ in range(num_iter)]

    if opts.rank != 0:
        return float('nan')

    return sum(fvd)/float(num_iter)
