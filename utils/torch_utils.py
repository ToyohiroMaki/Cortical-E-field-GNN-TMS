from typing import List
import copy

from torch import nn


__all__ = ['restore_model']


def restore_model(state_dict: dict, model: nn.Module):
    """
    remove ddp and compiled model flags.
    """
    state_dict = remove_keywords_from_key_of_dict(state_dict,
                                                  keys_to_remove=['_orig_mod.', '_orig_mod.module.', 'module.'])
    model.load_state_dict(state_dict, strict=True)


def remove_keywords_from_key_of_dict(state_dict: dict, keys_to_remove: List[str]):
    keys = copy.copy(list(state_dict.keys()))

    for k in keys:
        k_new = copy.copy(k)
        for kr in keys_to_remove:
            k_new = k_new.replace(kr, '', 1)

        if k != k_new:
            state_dict[k_new] = state_dict.pop(k)
    return state_dict
