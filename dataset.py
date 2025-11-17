import numpy as np
from torch.utils.data import Dataset

from utils.eeg_positions import EEGPositionsLeft
from utils.field_torch import MFieldFMM3D
from utils import mapping_numpy as mnp
from utils.intensity_normalization import intensity_normalization, get_scale_from_mni


def load_mri(subject_id):
    pass


def load_input_mesh(subject_id):
    pass


def load_gt_mesh(subject_id):
    pass


def load_ef(subject_id, position, angle):
    pass


def load_matsimnibs(subject_id, position, angle):
    pass


class MeshDatasetTMS(Dataset):
    def __init__(self):
        # sample
        self.subject_ids = ['1', '2', '3']
        self.positions = [EEGPositionsLeft.C1, EEGPositionsLeft.C3, EEGPositionsLeft.C5]
        self.angles = [0, 20, 40]

        self.field_module = MFieldFMM3D(kind='A')
        self.standard_scale, self.percentiles = get_scale_from_mni()

    def __len__(self):
        return len(self.subject_ids) * len(self.positions) * len(self.angles)

    def get_item(self, subject_id: str, position: EEGPositionsLeft, angle: float):
        data_dict = {'subject': subject_id, 'position': position, 'angle': angle}

        mri, affine = load_mri(subject_id)
        mri = intensity_normalization(mri, self.standard_scale, self.percentiles)
        mri = np.clip(mri / 100, 0, 1)
        mri = np.expand_dims(mri, axis=0)
        data_dict['mri'] = mri
        data_dict['affine'] = affine

        input_mesh = load_input_mesh(subject_id)
        data_dict['input_mesh'] = input_mesh

        gt_mesh = load_gt_mesh(subject_id)
        data_dict['gt_mesh'] = gt_mesh

        ef = load_ef(subject_id, position, angle)
        data_dict['ef'] = ef

        matsimnibs = load_matsimnibs(subject_id, position, angle)
        data_dict['matsimnibs'] = matsimnibs

        v_coil = mnp.voxel2coil(np.copy(input_mesh.vertices), matsimnibs=matsimnibs, affine=affine)
        a_field = self.field_module.compute_field(v_coil)
        a_field = mnp.coil2voxel(a_field, matsimnibs=matsimnibs, affine=affine, no_translation=True)
        data_dict['a_field'] = a_field

        return data_dict

    def __getitem__(self, index):
        angle_id = index % len(self.angles)
        tmp = index // len(self.angles)
        position_id = tmp % len(self.positions)
        tmp = tmp // len(self.positions)
        subject_idx = tmp % len(self.subject_ids)

        subject_id = self.subject_ids[subject_idx]
        position = self.positions[position_id]
        angle = self.angles[angle_id]
        return self.get_item(subject_id, position, angle)
