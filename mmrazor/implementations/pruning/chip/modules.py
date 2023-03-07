# Copyright (c) OpenMMLab. All rights reserved.

import torch


def reduced_1_row_norm(input, row_index, data_index):
    input[data_index, row_index, :] = torch.zeros(input.shape[-1])
    m = torch.norm(input[data_index, :, :], p='nuc').item()
    return m


def ci_score(conv_reshape: torch.Tensor):

    r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]],
                          device=conv_reshape.device)
    for i in range(conv_reshape.shape[0]):
        for j in range(conv_reshape.shape[1]):
            r1_norm[i, j] = reduced_1_row_norm(
                conv_reshape.clone(), j, data_index=i)

    ci = torch.zeros_like(r1_norm)

    for i in range(r1_norm.shape[0]):
        original_norm = torch.norm(
            torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        ci[i] = original_norm - r1_norm[i]

    # return shape: [batch_size, filter_number]
    return ci


def compute_ci(input: torch.Tensor):
    assert len(input.shape) == 3  # B C N
    return ci_score(input).mean(dim=0)  # C
