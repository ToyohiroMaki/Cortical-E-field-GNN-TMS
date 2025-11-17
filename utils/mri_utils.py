import nibabel as nib


__all__ = ['adjust_orientation']


def adjust_orientation(x):
    # https://github.com/nipy/nibabel/issues/1010
    orig_orient = nib.orientations.io_orientation(x.affine)
    trg_orient = nib.orientations.axcodes2ornt('RAS')
    transform = nib.orientations.ornt_transform(orig_orient, trg_orient)
    x = x.as_reoriented(transform)
    return x
