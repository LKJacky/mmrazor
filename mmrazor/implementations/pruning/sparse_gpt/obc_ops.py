# Copyright (c) OpenMMLab. All rights reserved.
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)
from .ops import SparseGptMixIn


class OBCMixin(SparseGptMixIn):

    @torch.no_grad()
    def prepare(self, columnslast=False):
        dev = self.weight_matrix.device
        if columnslast:
            perm = torch.arange(self.columns, device=dev)
            if len(self.weight.shape) == 4:
                perm = perm.reshape(list(self.weight.shape)[1:])
                perm = perm.permute([1, 2, 0])
                perm = perm.flatten()
        W = self.weight_matrix.clone()
        H = self.hessian.float().to(dev)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if columnslast:
            H = H[perm, :][:, perm]
            W = W[:, perm]
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=dev)
        if columnslast:
            return W, H, Hinv, Losses, perm
        return W, H, Hinv, Losses

    def invert(self, H):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            print('Hessian not full rank.')
            tmp = 1 * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare_iter(self, i1, parallel, W, Hinv1):
        dev = self.weight_matrix.device
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w = W[i1:i2, :]
        Hinv = Hinv1.unsqueeze(0).repeat((count, 1, 1))
        mask = torch.zeros_like(w).bool()
        rangecount = torch.arange(count, device=dev)
        idxcount = rangecount + i1
        return i2, count, w, Hinv, mask, rangecount, idxcount

    def prune(self,
              sparsity,
              prunen=0,
              prunem=0,
              blocksize=128,
              percdamp=0.01):
        if self.columns % 4 != 0:
            return
        self.prune24()

    def prune24(self):
        dev = self.weight_matrix.device
        parallel = 32
        n = 2
        m = 4
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(
                i1, parallel, W, Hinv1)

            buckets = torch.zeros((count, self.columns // m, 1), device=dev)

            tick = time.time()

            for zeros in range(1, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = w**2 / diag
                tmp = (buckets >= n).repeat((1, 1, m)).flatten(1)
                scores[mask | tmp] = float('inf')
                j = torch.argmin(scores, 1)
                Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                buckets[rangecount,
                        torch.div(j, m, rounding_mode='floor'), :] += 1
                if zeros == self.columns * n / m:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            Losses[i1:i2, :] /= 2
            w[mask] = 0
            W[i1:i2, :] = w

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        W = W[:, torch.argsort(perm)]
        self.weight.data = W.reshape(self.weight.shape)


class OBCLinear(DynamicLinear, OBCMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sparse_gpt_mix_in_init()

    @classmethod
    def convert_from(cls, module: nn.Linear) -> 'DynamicConv2d':
        if module.out_features < module.in_features:
            return module
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        dtype = next(module.parameters()).dtype
        new_module = new_module.to(dtype)

        return new_module


class OBCConv2d(DynamicConv2d, OBCMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sparse_gpt_mix_in_init()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        dtype = next(module.parameters()).dtype
        new_module = new_module.to(dtype)

        return new_module

    def format_input(self, input: torch.Tensor):
        # input B C H W
        input = F.unfold(
            input, self.kernel_size, padding=self.padding,
            stride=self.stride)  # B C D
        return input.transpose(-1, -2)
