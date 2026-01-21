import torch
from scipy.cluster.hierarchy import weighted

from torch import nn
from functorch import jvp, make_functional_with_buffers


def mask_multiply(coefs, mask, params):
    if params.ndim != 3:
        return (coefs * 0.).sum()
    if params.ndim == 1:
        return (coefs.sum(dim=-1) * params).sum(dim=0)  
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

    def get_optimized_params(self, coef=None):

        used_coef = coef if coef is not None else self.coef
        if self.partition is not None:
            dparams = [
                mask_multiply(
                    used_coef[:, i, ],
                    self.mask_mats[dp[0].shape],
                    torch.cat([d.unsqueeze(0) for d in dp], dim=0)
                )
                for i, dp in enumerate(zip(*self.dparams))
            ]
        elif self.blockwise:
            dparams = [
                sum([p * c[i] for p, c in zip(dp, used_coef)])
                for i, dp in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([p * c for p, c in zip(dp, used_coef)])
                for dp in zip(*self.dparams)
            ]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return new_params


class WeightedLinearizedModel(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:

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

        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)], start=torch.zeros_like(dp[0])) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)], start=torch.zeros_like(dp[0])) for dp in zip(*self.dparams)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp

    def get_optimized_params(self, coef=None):

        used_coef = self.coef if coef is None else coef
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, used_coef)])
                       for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, used_coef)])
                       for dp in zip(*self.dparams)]
        return [p + d for p, d in zip(self.params0, dparams)]

class NonlinearDVBASI(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True, partition=None, dparams=None, init_weights=None):
        super().__init__()
        func, params, buffers = make_functional_with_buffers(model)
        if init_weights is not None:
            for p, ext_p in zip(params, init_weights):
                p.data.copy_(ext_p.data)
        self.func = lambda p, b, x: func(p, b, x)
        self.params = nn.ParameterList(params)
        for p in self.params:
            p.requires_grad =False
        self.buffer = buffers
        self.train_preprocess = getattr(model, "train_preprocess", None)
        self.val_preprocess = getattr(model, "val_preprocess", None)
        self.cache_dir = getattr(model, "cache_dir", None)
        self.dparams = dparams if dparams is not None else [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise
        self.partition = partition
        if self.partition is not None:
            self.mask_mats = {}
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(params), self.partition))
        elif blockwise:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(params)))
        else:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors)))

    def _apply(self, fn):
        new_self = super()._apply(fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        if hasattr(new_self, "mask_mats"):
            new_self.mask_mats = {k: fn(v) for k, v in new_self.mask_mats.items()}
        return new_self

    def forward(self, x):
        if self.partition is not None:
            dparams = []
            for i, dp_group in enumerate(zip(*self.dparams)):
                shape_key = dp_group[0].shape
                mask_mat = self.mask_mats[shape_key]
                dp_tensor = torch.stack(dp_group, dim=0)
                dparam_combined = mask_multiply(self.coef[:, i, :], mask_mat, dp_tensor)
                dparams.append(dparam_combined)
        elif self.blockwise:
            dparams = [
                sum([dp * c[i] for dp, c in zip(dp_group, self.coef)])
                for i, dp_group in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([dp * c for dp, c in zip(dp_group, self.coef)])
                for dp_group in zip(*self.dparams)
            ]
        new_params = [p + d for p, d in zip(self.params, dparams)]
        return self.func(new_params, self.buffer, x)

    def get_optimized_params(self, coef=None):
        used_coef = self.coef if coef is None else coef
        if self.partition is not None:
            dparams = []
            for i, dp_group in enumerate(zip(*self.dparams)):
                shape_key = dp_group[0].shape
                mask_mat = self.mask_mats[shape_key]
                dp_tensor = torch.stack(dp_group, dim=0)
                dparam_combined = mask_multiply(used_coef[:, i, :], mask_mat, dp_tensor)
                dparams.append(dparam_combined)
        elif self.blockwise:
            dparams = [
                sum([dp * c[i] for dp, c in zip(dp_group, used_coef)])
                for i, dp_group in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([dp * c for dp, c in zip(dp_group, used_coef)])
                for dp_group in zip(*self.dparams)
            ]
        return [p + d for p, d in zip(self.params, dparams)]

class LinearizedDVBASI(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True, partition=None, dparams=None, init_weights=None) -> None:

        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        if init_weights is not None:
            for p, init_p in zip(self.params0, init_weights):
                p.data.copy_(init_p.data)
        self.dparams = dparams if dparams is not None else [
            [tv.vector[k] for k in tv.vector if k.startswith('model.params.')]
            for tv in task_vectors
        ]

        self.blockwise = blockwise
        self.partition = partition

        if self.partition is not None:
            self.mask_mats = {}
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params0), self.partition))
        elif blockwise:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
        else:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors)))

    def _apply(self, fn):
        new_self = super()._apply(fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        if self.partition is not None and hasattr(new_self, "mask_mats"):
            new_self.mask_mats = {k: fn(v) for k, v in new_self.mask_mats.items()}
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.partition is not None:
            dparams = []
            for i, dp_group in enumerate(zip(*self.dparams)):
                shape_key = dp_group[0].shape
                mask_mat = self.mask_mats[shape_key]
                dp_tensor = torch.stack(dp_group, dim=0)
                dparam_combined = mask_multiply(self.coef[:, i, :], mask_mat, dp_tensor)
                dparams.append(dparam_combined)
        elif self.blockwise:
            dparams = [
                sum([dp * c[i] for dp, c in zip(dp_group, self.coef)])
                for i, dp_group in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([dp * c for dp, c in zip(dp_group, self.coef)])
                for dp_group in zip(*self.dparams)
            ]

        out, dp_out = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp_out

    def get_optimized_params(self, coef=None):
        used_coef = self.coef if coef is None else coef
        if self.partition is not None:
            dparams = []
            for i, dp_group in enumerate(zip(*self.dparams)):
                shape_key = dp_group[0].shape
                mask_mat = self.mask_mats[shape_key]
                dp_tensor = torch.stack(dp_group, dim=0)
                dparam_combined = mask_multiply(used_coef[:, i, :], mask_mat, dp_tensor)
                dparams.append(dparam_combined)
        elif self.blockwise:
            dparams = [
                sum([dp * c[i] for dp, c in zip(dp_group, used_coef)])
                for i, dp_group in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([dp * c for dp, c in zip(dp_group, used_coef)])
                for dp_group in zip(*self.dparams)
            ]
        return [p + d for p, d in zip(self.params0, dparams)]


