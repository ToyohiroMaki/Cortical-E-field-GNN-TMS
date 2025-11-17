import torch
from torch import Tensor


__all__ = [
    'voxel2subject', 'subject2voxel', 'coil2subject', 'subject2coil', 'voxel2coil', 'coil2voxel',
]


def affine_transform(x: Tensor, t: Tensor, no_translation: bool = False) -> torch.Tensor:
    if no_translation:
        y = t[:3, :3].mm(x.T)
        return y.T
    else:
        y = t[:3, :3].mm(x.T) + t[:3, -1][:, None]
        return y.T


def voxel2subject(x: Tensor, affine: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(affine, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    return affine_transform(x, affine, no_translation=no_translation)


def subject2voxel(x: Tensor, affine: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(affine, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    t = torch.linalg.inv(affine)
    return affine_transform(x, t, no_translation=no_translation)


def coil2subject(x: Tensor, matsimnibs: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(matsimnibs, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    return affine_transform(x, matsimnibs, no_translation=no_translation)


def subject2coil(x: Tensor, matsimnibs: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(matsimnibs, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    t = torch.linalg.inv(matsimnibs)
    return affine_transform(x, t, no_translation=no_translation)


def voxel2coil(x: Tensor, matsimnibs: Tensor, affine: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(matsimnibs, Tensor) and isinstance(affine, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    t = torch.mm(torch.linalg.inv(matsimnibs), affine)
    return affine_transform(x, t, no_translation=no_translation)


def coil2voxel(x: Tensor, matsimnibs: Tensor, affine: Tensor, no_translation: bool = False):
    assert isinstance(x, Tensor) and isinstance(matsimnibs, Tensor) and isinstance(affine, Tensor)
    assert x.ndim == 2 and x.size(1) == 3
    t = torch.mm(torch.linalg.inv(affine), matsimnibs)
    return affine_transform(x, t, no_translation=no_translation)
