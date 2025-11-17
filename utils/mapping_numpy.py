import numpy as np


__all__ = [
    'voxel2subject', 'subject2voxel', 'coil2subject', 'subject2coil', 'voxel2coil', 'coil2voxel',
]


def affine_transform(x: np.ndarray, t: np.ndarray, no_translation: bool = False) -> np.ndarray:
    if no_translation:
        y = np.matmul(t[:3, :3], x.T)
        return np.ascontiguousarray(y.T)
    else:
        y = np.matmul(t[:3, :3], x.T) + t[:3, -1][:, None]
        return np.ascontiguousarray(y.T)


def voxel2subject(x: np.ndarray, affine: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(affine, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    return affine_transform(x, affine, no_translation=no_translation)


def subject2voxel(x: np.ndarray, affine: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(affine, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    t = np.linalg.inv(affine)
    return affine_transform(x, t, no_translation=no_translation)


def coil2subject(x: np.ndarray, matsimnibs: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(matsimnibs, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    return affine_transform(x, matsimnibs, no_translation=no_translation)


def subject2coil(x: np.ndarray, matsimnibs: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(matsimnibs, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    t = np.linalg.inv(matsimnibs)
    return affine_transform(x, t, no_translation=no_translation)


def voxel2coil(x: np.ndarray, matsimnibs: np.ndarray, affine: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(matsimnibs, np.ndarray) and isinstance(affine, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    t = np.matmul(np.linalg.inv(matsimnibs), affine)
    return affine_transform(x, t, no_translation=no_translation)


def coil2voxel(x: np.ndarray, matsimnibs: np.ndarray, affine: np.ndarray, no_translation: bool = False):
    assert isinstance(x, np.ndarray) and isinstance(matsimnibs, np.ndarray) and isinstance(affine, np.ndarray)
    assert x.ndim == 2 and x.shape[1] == 3
    t = np.matmul(np.linalg.inv(affine), matsimnibs)
    return affine_transform(x, t, no_translation=no_translation)
