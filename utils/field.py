import os

import numpy as np
import fmm3dpy


__all__ = ['read_ccd', 'A_from_dipoles', 'B_from_dipoles', 'MFieldFMM3D']


def read_ccd(file: str):
    """
    This function is taken from SIMNIBS software (v4.0.1).
    (simnibs/simnibs/simulation/coil_numpy.py)

    Returns
    ----------
    [pos, m]: list
        position and moment of dipoles
    """
    assert file.split('.')[-1] == 'ccd'

    ccd_file = np.loadtxt(file, skiprows=2)

    # if there is only 1 dipole, loadtxt return as array of the wrong shape
    if (len(np.shape(ccd_file)) == 1):
        a = np.zeros([1, 6])
        a[0, 0:3] = ccd_file[0:3]
        a[0, 3:] = ccd_file[3:]
        ccd_file = a

    return ccd_file[:, 0:3], ccd_file[:, 3:]


def A_from_dipoles(d_moment, d_position, target_positions, eps=1e-3, direct='auto'):
    '''
    This function is taken from SIMNIBS software (v4.0.1).
    (simnibs/simnibs/simulation/coil_numpy.py)

    Get A field from dipoles using FMM3D

    Parameters
    ----------
    d_moment : ndarray
        dipole moments (Nx3).
    d_position : ndarray
        dipole positions (Nx3).
    target_positions : ndarray
        positions for which to calculate the A field.
    eps : float
        Precision. The default is 1e-3
    direct : bool
        Set to true to force using direct (naive) approach or False to force use of FMM.
        If set to auto direct method is used for less than 300 dipoles which appears to be faster in these cases.
        The default is 'auto'

    Returns
    -------
    A : ndarray
        A field at points (M x 3) in Tesla*meter.

    '''
    #if set to auto use direct methods if # dipoles less than 300
    if direct=='auto':
        if d_moment.shape[0]<300:
            direct = True
        else:
            direct = False
    if direct is True:
        out = fmm3dpy.l3ddir(charges=d_moment.T, sources=d_position.T,
                  targets=target_positions.T, nd=3, pgt=2)
    elif direct is False:
        #use fmm3dpy to calculate expansion fast
        out = fmm3dpy.lfmm3d(charges=d_moment.T, eps=eps, sources=d_position.T,
                  targets=target_positions.T, nd=3, pgt=2)
    else:
        print('Error: direct flag needs to be either "auto", True or False')
    A = np.empty((target_positions.shape[0], 3), dtype=float)
    #calculate curl
    A[:, 0] = (out.gradtarg[1][2] - out.gradtarg[2][1])
    A[:, 1] = (out.gradtarg[2][0] - out.gradtarg[0][2])
    A[:, 2] = (out.gradtarg[0][1] - out.gradtarg[1][0])
    #scale
    A *= -1e-7
    return A


def B_from_dipoles(d_moment, d_position, target_positions, eps=1e-3, direct='auto'):
    '''
    This function is taken from SIMNIBS software (v4.0.1).
    (simnibs/simnibs/simulation/coil_numpy.py)

    Get B field from dipoles using FMM3D

    Parameters
    ----------
    d_moment : ndarray
        dipole moments (Nx3).
    d_position : ndarray
        dipole positions (Nx3).
    target_positions : ndarray
        position for which to calculate the B field.
    eps : float
        Precision. The default is 1e-3
    direct : bool
        Set to true to force using direct (naive) approach or False to force use of FMM.
        If set to auto direct method is used for less than 300 dipoles which appears to be faster i these cases.
        The default is 'auto'

    Returns
    -------
    B : ndarray
        B field at points (M x 3) in Tesla.

    '''
    #if set to auto use direct methods if # dipoles less than 300
    if direct=='auto':
        if d_moment.shape[0]<300:
            direct = True
        else:
            direct = False
    if direct is True:
        out = fmm3dpy.l3ddir(dipvec=d_moment.T, sources=d_position.T,
                  targets=target_positions.T, nd=1, pgt=2)
    elif direct is False:
        out = fmm3dpy.lfmm3d(dipvec=d_moment.T, eps=eps, sources=d_position.T,
                  targets=target_positions.T, nd=1, pgt=2)
    else:
        print('Error: direct flag needs to be either "auto", True or False')
    B = out.gradtarg.T
    B *= -1e-7
    return B


class MFieldFMM3D:
    def __init__(self, kind: str):
        super(MFieldFMM3D, self).__init__()
        assert kind in ['B', 'A']
        self.kind = kind

        ccd_path = os.path.join(os.path.dirname(__file__), 'Magstim_70mm_Fig8.ccd')
        d_position, d_moment = read_ccd(ccd_path)
        d_position *= 1e3

        self.d_position = d_position
        self.d_moment = d_moment

    def compute_field(self, points: np.ndarray) -> np.ndarray:
        """
        :param points: point of coil coordinates
        :return: field at specified points
        """
        if self.kind == 'B':
            f = B_from_dipoles
        else:
            f = A_from_dipoles

        field = f(self.d_moment, self.d_position, points)
        field = np.reshape(field, (-1, 3))
        return field
