# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from einops import rearrange

import torch.nn.functional as F

from src.training.diffaugment import DiffAugment


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()


class OurStyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, diffaugment='', augment_pipe=None,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.diffaugment = diffaugment
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.dist1 = torch.distributions.beta.Beta(2., 1., validate_args=None)
        self.dist2 = torch.distributions.beta.Beta(1., 2., validate_args=None)

    def run_G(self, z, c, sync):
        batch_size = z.size(0)
        Ts = torch.cat([self.dist1.sample((batch_size, 1, 1, 1, 1)),
                               self.dist2.sample((batch_size, 1, 1, 1, 1))], dim=1).to(self.device)
        Ts = torch.cat([Ts.min(dim=1, keepdim=True)[0], Ts.max(dim=1, keepdim=True)[0]], dim=1)
        Ts = rearrange(Ts, 'b t c h w -> (b t) c h w')

        z_motion = torch.randn(batch_size, 512).to(z.device)

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                        torch.full_like(cutoff, ws.shape[1])
                    )
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, Ts, z_motion)

        return img, ws, Ts

    def run_D(self, img, c, sync):
        if self.diffaugment:
            img = torch.cat([
                    rearrange(DiffAugment(img, policy=self.diffaugment), '(b t) c h w -> b (t c) h w', t=2),
                    img[:, 6:7]], dim=1
                  )

        with misc.ddp_sync(self.D, sync):
            logits, img_logits, temp_imgs = self.D(img, c)
        return logits, img_logits, temp_imgs

    def convert(self, img, Ts):
        img = rearrange(img, '(b t) c h w -> b (t c) h w', t=2)
        Ts = Ts.view(-1, 2, 1, 1)[:, 1:2] - Ts.view(-1, 2, 1, 1)[:, 0:1]
        Ts = Ts * torch.ones_like(img[:, 0:1])
        img = torch.cat([img, Ts], dim=1)

        return img

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, _gen_Ts = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_img = self.convert(gen_img, _gen_Ts)
                gen_logits, gen_img_logits, _ = self.run_D(gen_img, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gimg_main = torch.nn.functional.softplus(-gen_img_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/Iloss', loss_Gimg_main)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean() + loss_Gimg_main.mean()).mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, _gen_Ts = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, _gen_Ts = self.run_G(gen_z, gen_c, sync=False)
                gen_img = self.convert(gen_img, _gen_Ts)
                gen_logits, gen_img_logits, _ = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                loss_Dimg_gen = torch.nn.functional.softplus(gen_img_logits)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen.mean() + loss_Dimg_gen.mean()).mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                #print(real_img.size())
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_img_logits, temp_imgs = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_Dimg_real = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dimg_real = torch.nn.functional.softplus(-real_img_logits)

                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/Iloss', loss_Dimg_gen + loss_Dimg_real)

                loss_Dr1 = 0
                loss_Dimg_r1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_img_grads = torch.autograd.grad(outputs=[real_img_logits.sum()], inputs=[temp_imgs], create_graph=True, only_inputs=True)[0]

                    r1_penalty = r1_grads.square().sum([1,2,3])
                    r1_img_penalty = r1_img_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dimg_r1 = r1_img_penalty * (self.r1_gamma / 2)

                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                ((real_logits * 0 + loss_Dreal + loss_Dr1).mean() +
                 (real_img_logits * 0 + loss_Dimg_real + loss_Dimg_r1).mean()).mul(gain).backward()
