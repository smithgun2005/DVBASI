"""Models with learnable weights on task vectors

Fred Zhang <frederic.zhang@adelaide.edu.au>
Paul Albert <paul.albert@adelaide.edu.au>

Australian Institute for Machine Learning
"""

import torch

from torch import nn
from functorch import jvp, make_functional_with_buffers


def mask_multiply(coefs, mask, params):
    if params.ndim != 3:
        return (coefs * 0.).sum()
    if params.ndim == 1:
        return (coefs.sum(dim=-1) * params).sum(dim=0)  # Classic block wise for 1-dim parameters
    if params.ndim == 2:
        coef_mask = torch.einsum('ij,jk->ik', coefs, mask.to(coefs))
        return torch.einsum('ik,ik->k', coef_mask, params)
    if params.ndim == 5:  # Conv layer
        coef_mask = torch.einsum('ij,jdkcb->idkcb', coefs, mask.to(coefs))
        return torch.einsum('idkcb,idkcb->dkcb', coef_mask, params)

    coef_mask = torch.einsum('ij,jbk->ibk', coefs.to(mask), mask)
    return torch.einsum('ibk,ibk->bk', coef_mask, params)


class WeightedImageEncoder(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True, partition=None,doubleblockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors

        Parameter:
        ----------
        model: nn.Module
            CLIP image encoder model.
        task_vectors: List[NonLinearTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        func, params, self.buffer = make_functional_with_buffers(model)
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise
        self.partition = partition
        if self.partition is not None:
            self.mask_mats = {}
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params), self.partition))
            for p in self.params:
                mask = torch.randint(self.partition, p.shape)
                self.mask_mats[p.shape] = torch.nn.Parameter(torch.nn.functional.one_hot(mask).moveaxis(-1, 0).half(),
                                                             requires_grad=False)
        elif blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), ))




    def _apply(self, fn):
        """Override method to relocate buffer list

        NOTE: This function signature is for PyTorch 1.13.1.
        Newer verions have added another optional argument `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        if hasattr(self, 'mask_mats'):
            new_self.mask_mats = {k: fn(v) for k, v in new_self.mask_mats.items()}
        return new_self

    def train(self, mode=True):
        super().train(mode)

    def forward(self, x) -> torch.Tensor:
        if self.partition is not None:
            dparams = [mask_multiply(self.coef[:, i, ], self.mask_mats[dp[0].shape],
                                     torch.cat([d.unsqueeze(0) for d in dp], dim=0)) for i, dp in
                       enumerate(zip(*self.dparams))]
        elif self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, self.buffer, x)

class SuperWeightedImageEncoder(nn.Module):
    def __init__(self, model, task_vectors, Atlass_lamda,  num_blocks: int = 10,) -> None:
        super().__init__()
        self.model = model
        self.task_vectors = task_vectors
        func, params, self.buffer = make_functional_with_buffers(model)
        self.func = lambda p, b, x: func(p, b, x)
        self.params = nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]

        block_sizes = []
        base_size = self.num_params // self.num_blocks  # ?num_paramsga    感觉可以用atlas_lamda的某个维度去写
        remainder = self.num_params % self.num_blocks

        start_idx = 0
        for i in range(num_blocks):
            size_i = base_size + (1 if i < remainder else 0)
            block_sizes.append(size_i)

        self.block_slices = []
        for size_i in block_sizes:
            end_idx = start_idx + size_i
            self.block_slices.append((start_idx, end_idx))
            start_idx = end_idx

        base_one = torch.ones_like(Atlass_lamda)

        # todo: load checkpoints of atlas
        delta_inital=Atlass_lamda-base_one
        self.register_buffer("base_one", base_one)
        self.delta_coef=torch.nn.Parameter(delta_inital)

        for (st, ed) in self.block_slices:
            # shape = (num_tasks, block_size)
            block_data = self.delta_coef[:, st:ed]
            param_block = nn.Parameter(block_data)
            self.delta_blocks.append(param_block)

    def  forward(self,x):
        delta_list = []
        for i, (st, ed) in enumerate(self.block_slices):
            delta_list.append(self.delta_blocks[i])  # shape = (num_tasks, ed-st)
        # 沿着列 concat
        delta_full = torch.cat(delta_list, dim=1)  # shape=(num_tasks, num_params)

        # todo: 1. get coefficient
        final_coef=self.base_one+delta_full

        # todo 2: multiply task vector p就是任务向量
        dparams = [
            sum([p * final_coef[t_i, i] for t_i, p in enumerate(dp)])
            for i, dp in enumerate(zip(*self.dparams))
        ]

        # todo 3: add parameters dp是原来的参数
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, self.buffer, x)


class WeightedLinearizedModel(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors for linearised models

        Parameters:
        -----------
        model: nn.Module
            Linearised model using first-order Taylor expansion.
        task_vectors: List[LinearizedTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), ))

    def _apply(self, fn):
        """Override method to relocate buffer list
        NOTE: This function signature is for PyTorch 1.13.1.
        Newer verions have added another optional argument `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp
