import os
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
import cc3d
import nibabel as nib


__all__ = ['intensity_normalization', 'get_scale_from_mni', 'get_foreground_region']


def intensity_normalization(x: np.ndarray, standard_scale: np.ndarray, percentiles: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
    assert standard_scale.ndim == percentiles.ndim == 1
    assert len(standard_scale) == len(percentiles)
    assert np.all(x >= 0)

    percentiles = np.copy(percentiles)
    percentiles[-1] = 99

    mask = get_foreground_region(x) if mask is None else mask
    landmarks = np.percentile(x[mask].flatten(), percentiles)

    f = interp1d(landmarks, standard_scale, kind='linear', fill_value='extrapolate')
    y = f(x.flatten())
    y = np.reshape(y, x.shape)
    return y


def get_scale_from_mni():
    mni = nib.load(os.path.join(__file__, 'sample/T1.nii'))
    mni = mni.get_fdata()
    mask = get_foreground_region(mni)
    """
    L G. Nyul et al., New Variants of a Method of MRI Scale Standardization
    """
    percentiles = np.array([0] + np.arange(0, 100, 10)[1:].tolist() + [100])
    landmark = np.percentile(mni[mask], percentiles)
    return landmark, percentiles


def get_foreground_region(x: np.ndarray, th: float = None) -> np.ndarray:
    """
    L G. Nyul et al., New Variants of a Method of MRI Scale Standardization
    """
    assert x.ndim == 3
    th = x.mean() if th is None else th
    mask = (x > th)

    i_num = 2
    mask = np.pad(mask, pad_width=i_num, mode='constant', constant_values=True)
    mask = binary_opening(mask, structure=generate_binary_structure(3, 3), iterations=i_num)
    mask = binary_closing(mask, structure=generate_binary_structure(3, 3), iterations=i_num)
    mask = mask[i_num:-i_num, i_num:-i_num, i_num:-i_num]

    mask = cc3d.dust(mask, threshold=100000, connectivity=26, in_place=False)
    return mask
