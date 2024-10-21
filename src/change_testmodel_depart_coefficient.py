from __future__ import annotations

import numpy as np
from run_m3dis import synthesize_spectra
import importlib.util
import sys
import os
import shutil

# Created by storm at 21.10.24

def import_module_from_path(module_name, file_path):
    """
    Dynamically imports a module or package from a given file path.

    Parameters:
    module_name (str): The name to assign to the module.
    file_path (str): The file path to the module or package.

    Returns:
    module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Module spec not found for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

module_path = os.path.join("/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/", "m3dis/__init__.py")
m3dis_python_module = import_module_from_path("m3dis", module_path)


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


def rerun_new_depart_coefficient(m3dis_path, m3d_path_run, atmo_path, atom_path, atom_abund, path_precomputed_depart, min_x, max_x, verbose=True):
    synthesize_spectra(m3dis_path, m3d_path_run, atmo_path, atom_path, atom_abund, 0.01, 0, False,
                       "", 10, True, verbose, path_precomputed_depart)
    xx_lte, yy_lte, xx_nlte, yy_nlte = get_synthetic_spectra(m3dis_python_module, m3d_path_run, min_x, max_x)
    shutil.rmtree(m3d_path_run)
    return xx_lte, yy_lte, xx_nlte, yy_nlte

def get_xx_yy(run, min_x, max_x, LTE=False, norm=True):
    xx, mask = run.get_xx(lam=run.lam, xmin=min_x, xmax=max_x)
    yy, cont = run.get_yy(mask=mask, norm=norm, LTE=LTE)
    return xx, yy

def get_synthetic_spectra(m3dis, run_path, min_x, max_x):
    run = m3dis.read(run_path)
    xx_nlte, yy_nlte = get_xx_yy(run, min_x, max_x, norm=True, LTE=False)
    xx_lte, yy_lte = get_xx_yy(run, min_x, max_x, norm=True, LTE=True)
    return xx_lte, yy_lte, xx_nlte, yy_nlte