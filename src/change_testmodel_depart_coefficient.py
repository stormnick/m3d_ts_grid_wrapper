from __future__ import annotations

import numpy as np
from run_m3dis import synthesize_spectra

# Created by storm at 21.10.24


def change_depart_coefficient(sfolder: str, lower_level: np.ndarray[float], upper_level: np.ndarray[float], atmo_dims: int, atom_levels: int):
    file = "atom_patch_001000.bin"
    dtype = "<f4"
    dims = (1, 1, atmo_dims, atom_levels, 2, 1)

    depart_file = np.memmap(sfolder + file, dtype=dtype, mode="r+", shape=dims, order="F")

    # dims are atmo depth, levels, lte/depart
    #lte = np.squeeze(test[:, :, :, :, 0, :])
    depart = np.squeeze(depart_file[:, :, :, :, 1, :])

    # set all depart to ones
    new_depart = np.ones_like(depart)

    new_depart[:, 1] = lower_level  # lower level
    new_depart[:, 4] = upper_level  # upper level

    # new_depart = np.random.rand(*depart.shape).astype(dtype)  # Example new data

    depart_file[0, 0, :, :, 1, 0] = new_depart
    depart_file.flush()


def rerun_new_depart_coefficient(m3dis_path, m3d_path_run, atmo_path, atom_path, atom_abund, path_precomputed_depart, verbose=False):
    synthesize_spectra(m3dis_path, m3d_path_run, atmo_path, atom_path, atom_abund, 0.01, 0, False,
                       "", 10, True, verbose, path_precomputed_depart)
    return m3d_path_run