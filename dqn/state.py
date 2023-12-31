from typing import Type, Dict

import torch
import torch.nn.functional as F
from torch import nn

from .base import _GLOBAL_DEVICE
from .model import Network, CNN, CNN3, CNN2

_STATE_TRANS: Dict[str, Type['BaseStateTransform']] = {}


def _get_state_trans(name):
    return _STATE_TRANS[name]


def _register_state_trans(name):
    def _decorator(cls: Type['BaseStateTransform']):
        _STATE_TRANS[name] = cls
        return cls

    return _decorator


class BaseStateTransform:
    def __init__(self, state_dims, device=_GLOBAL_DEVICE):
        self.state_dims = state_dims
        self.device = device

    def state_trans(self, state: int):
        raise NotImplementedError

    def create_model(self, action_dims: int) -> nn.Module:
        raise NotImplementedError


class SimpleStateTransform(BaseStateTransform):

    def _input_dims(self):
        raise NotImplementedError

    @property
    def input_dims(self):
        return self._input_dims()

    def _trans(self, state: int):
        raise NotImplementedError

    def state_trans(self, state: int):
        new_state = self._trans(state)
        assert new_state.shape[-1] == self._input_dims()
        return new_state.float().to(self.device)

    def create_model(self, action_dims: int) -> nn.Module:
        raise NotImplementedError


@_register_state_trans('standalone')
class StandaloneStateTransform(SimpleStateTransform):
    def _input_dims(self):
        return 1

    def _trans(self, state: int):
        return torch.tensor([state]).float()

    def create_model(self, action_dims: int) -> nn.Module:
        return Network(self._input_dims(), action_dims)


@_register_state_trans('one_hot')
class OneHotStateTransform(SimpleStateTransform):
    def _input_dims(self):
        return self.state_dims

    def _trans(self, state: int):
        return F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()

    def create_model(self, action_dims: int) -> nn.Module:
        return Network(self._input_dims(), action_dims)


@_register_state_trans('one_hot_xy')
class OneHotStateTransform(SimpleStateTransform):
    def _input_dims(self):
        return self.state_dims + 2

    def _trans(self, state: int):
        oh = F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()
        xy = torch.tensor([state // 12, state % 12]).float()
        return torch.cat([oh, xy]).float()

    def create_model(self, action_dims: int) -> nn.Module:
        return Network(self._input_dims(), action_dims)


class _BaseCNNTransform(BaseStateTransform):
    def __init__(self, state_dims, device=_GLOBAL_DEVICE, **kwargs):
        BaseStateTransform.__init__(self, state_dims, device)
        self.model_kwargs = kwargs

    def state_trans(self, state: int):
        oh = F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()
        oh = oh.reshape(4, 12).unsqueeze(0)
        return oh.to(self.device)

    def create_model(self, action_dims: int) -> nn.Module:
        return CNN((4, 12), action_dims, **self.model_kwargs)


@_register_state_trans('cnn')
class CNNTransform(_BaseCNNTransform):
    def __init__(self, state_dims, device=_GLOBAL_DEVICE):
        _BaseCNNTransform.__init__(
            self, state_dims, device,
            kernel_size=3, psize=1,
        )


@_register_state_trans('cnn_k5_p2')
class CNNTransform(_BaseCNNTransform):
    def __init__(self, state_dims, device=_GLOBAL_DEVICE):
        _BaseCNNTransform.__init__(
            self, state_dims, device,
            kernel_size=5, psize=2,
        )


@_register_state_trans('cnn_k7_p2')
class CNNTransform(_BaseCNNTransform):
    def __init__(self, state_dims, device=_GLOBAL_DEVICE):
        _BaseCNNTransform.__init__(
            self, state_dims, device,
            kernel_size=7, psize=2,
        )


@_register_state_trans('cnn2')
class CNN3Transform(BaseStateTransform):
    def state_trans(self, state: int):
        oh = F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()
        oh = oh.reshape(4, 12).unsqueeze(0)
        return oh.to(self.device)

    def create_model(self, action_dims: int) -> nn.Module:
        return CNN2(action_dims)


@_register_state_trans('cnn3')
class CNN3Transform(BaseStateTransform):
    def state_trans(self, state: int):
        oh = F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()
        oh = oh.reshape(4, 12).unsqueeze(0)
        return oh.to(self.device)

    def create_model(self, action_dims: int) -> nn.Module:
        return CNN3(action_dims)
