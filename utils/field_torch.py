import os

import numpy as np

from .field import read_ccd, B_from_dipoles, A_from_dipoles


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
        :return: magnetic field (or potential) at specified points
        """
        if self.kind == 'B':
            f = B_from_dipoles
        else:
            f = A_from_dipoles

        field = f(self.d_moment, self.d_position, points)
        field = np.reshape(field, (-1, 3))
        return field
