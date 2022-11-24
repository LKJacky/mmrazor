# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch


def normal_equation(X: torch.Tensor, Y: torch.Tensor):
    '''
    X: M * N
    Y: M * K
    R: N * K
    X@R=Y
    '''
    try:
        R = (X.T @ X).inverse() @ X.T @ Y
    except Exception:
        R = (X.T @ X).pinverse() @ X.T @ Y
    return R


def simple_back_cssp(m: torch.Tensor, k=0):
    '''
    m: M * N
    '''
    sort = []
    loss = []
    _, N = m.shape
    mask = m.new_ones([N], dtype=torch.bool)

    for _ in range(N - k):
        ls = torch.tensor([torch.inf] * N, dtype=m.dtype)
        for j in range(N):
            if mask[j] == 0:
                A = None
                pass
            else:
                mask[j] = 0
                mm = m[:, mask]
                A = normal_equation(mm, m)
                l_ = (m - mm @ A).norm()
                ls[j] = l_
                mask[j] = 1

        k = ls.argmin()
        mask[k] = 0
        sort.append(k)
        loss.append(ls[k])
    sort = torch.tensor(sort)
    loss = torch.tensor(loss)
    return sort, loss


# quick_back_cssp


def get_A_L(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, N = m.shape
    mask = m.new_ones([N], dtype=torch.bool)
    A = []
    L = []
    for i in range(N):
        mask[i] = 0
        mm = m[:, mask]
        mi = m[:, i].unsqueeze(-1)
        Ai = normal_equation(mm, mi)
        Li = mi - mm @ Ai
        L.append(Li)
        Ai = Ai.flatten()
        Ai = torch.tensor(
            Ai[:i].tolist() + [0.0] + Ai[i:].tolist(),
            dtype=Ai.dtype,
            device=mm.device).unsqueeze(-1)
        A.append(Ai)
        mask[i] = 1
    A = torch.cat(A, dim=-1)
    L = torch.cat(L, dim=-1)
    return A, L


def update_L(L: torch.Tensor, A: torch.Tensor, j: int):
    L_j = L[:, j].unsqueeze(-1)
    Aj = A[j].unsqueeze(0)
    A_j = A[:, j].unsqueeze(-1)
    L1 = L_j @ Aj + L
    L1 = L1 / ((1 - Aj.flatten() * A_j.flatten()).unsqueeze(0))
    return L1


def update_G(G: torch.Tensor, A: torch.Tensor, j: int):
    Aj = A[j].unsqueeze(0)
    A_j = A[:, j].unsqueeze(-1)
    G1 = Aj.T@Aj*G[j, j] \
        + Aj.T@G[j].unsqueeze(0) \
        + G[:, j].unsqueeze(-1)@Aj \
        + G
    d = (1 - Aj * A_j.T).T @ (1 - Aj * A_j.T)
    G1 = G1 / d

    return G1


def update_A(A: torch.Tensor, j: int):
    Aj = A[j].unsqueeze(0)
    A_j = A[:, j].unsqueeze(-1)
    A1 = A + A_j @ Aj
    A1 = A1 / (1 - Aj * A_j.T)
    A1[j] = 0
    A1.fill_diagonal_(0)
    return A1


def get_LOSS(A: torch.Tensor, G: torch.Tensor,
             select: torch.Tensor) -> torch.Tensor:
    LOSS = G.diagonal()*A[:, ~select].square().sum(dim=1) \
        + G.diagonal()[~select].sum() \
        + 2*(A[:, ~select]*G[~select].T).sum(dim=1) \
        + G.diagonal()
    LOSS = LOSS.sqrt()
    LOSS[~select] = torch.inf
    return LOSS


def quick_back_cssp(m: torch.Tensor, k=0):

    sort = []
    loss = []
    _, N = m.shape
    A, L = get_A_L(m)
    G = L.T @ L
    mask = m.new_ones([N], dtype=torch.bool)
    for i in range(N - k):

        LOSS = get_LOSS(A, G, mask)
        LOSS[~mask] = torch.inf
        k = LOSS.argmin()
        mask[k] = 0
        sort.append(k)
        loss.append(LOSS[k])
        # update
        G = update_G(G, A, k)
        A = update_A(A, k)
    sort = torch.tensor(sort)
    loss = torch.tensor(loss)
    return sort, loss
