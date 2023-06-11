from typing import Type, Dict

import torch
import torch.nn.functional as F

from .base import _GLOBAL_DEVICE

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


@_register_state_trans('standalone')
class StandaloneStateTransform(BaseStateTransform):
    def _input_dims(self):
        return 1

    def _trans(self, state: int):
        return torch.tensor([state]).float()


@_register_state_trans('one_hot')
class OneHotStateTransform(BaseStateTransform):
    def _input_dims(self):
        return self.state_dims

    def _trans(self, state: int):
        return F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()


@_register_state_trans('one_shot_xy')
class OneHotStateTransform(BaseStateTransform):
    def _input_dims(self):
        return self.state_dims + 2

    def _trans(self, state: int):
        oh = F.one_hot(torch.tensor(state), num_classes=self.state_dims).float()
        xy = torch.tensor([state // 12, state % 12]).float()
        return torch.cat([oh, xy]).float()
