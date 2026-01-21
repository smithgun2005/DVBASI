
import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers

from src.modeling import ImageEncoder
from src.utils import DotDict


class LinearizedModel(nn.Module):


    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, buffers, x: func0(params, buffers, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        for p in self.params0:
            p.requires_grad = False

        for p in self.params:
            p.requires_grad = True

    def _apply(self, fn):

        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        return new_self

    def __call__(self, x) -> torch.Tensor:
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class LinearizedImageEncoder(abc.ABC, nn.Module):

    def __init__(
        self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
    ):
        super().__init__()
        if image_encoder is None:
            image_encoder = ImageEncoder(args, keep_lang)
        if init_encoder is None:
            init_encoder = image_encoder


        self.train_preprocess = image_encoder.train_preprocess
        self.val_preprocess = image_encoder.val_preprocess
        self.cache_dir = image_encoder.cache_dir

        self._model_name = self._get_name(args.model)
        self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

    def _get_name(self, model_name):
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", "")
        return model_name

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def save(self, filename):
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):

        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")

        args = DotDict({"model": state_dict["model_name"]})
        taylorized_encoder = cls(args)

        state_dict.pop("model_name")
        taylorized_encoder.load_state_dict(state_dict)
        return taylorized_encoder
