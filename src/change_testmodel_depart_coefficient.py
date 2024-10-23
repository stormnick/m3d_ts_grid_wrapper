from __future__ import annotations

import numpy as np
from run_m3dis import synthesize_spectra
import importlib.util
import sys
import os
import shutil
from auxfuncs import compute_tau_scale, interp_tau, rotate_array

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
    xx_lte, yy_lte, xx_nlte, yy_nlte, ltau_lte, cf_lte, ltau_nlte, cf_nlte = get_synthetic_spectra(m3dis_python_module, m3d_path_run, min_x, max_x)
    shutil.rmtree(m3d_path_run)
    return xx_lte, yy_lte, xx_nlte, yy_nlte, ltau_lte, cf_lte, ltau_nlte, cf_nlte

def get_xx_yy(run, min_x, max_x, LTE=False, norm=True):
    xx, mask = run.get_xx(lam=run.lam, xmin=min_x, xmax=max_x)
    yy, cont = run.get_yy(mask=mask, norm=norm, LTE=LTE)
    return xx, yy

def get_cntrbf(self, df=None, ang=None, imu=None, LTE=False, mean=False, fslice=None, norm=False):
    # -----------------------------------------------------------------
    cntrbf_lines = np.array(self.run.nml['line_mask']['cntrbf_lines']).ravel() - 1
    if not self.kr in cntrbf_lines:
        raise ValueError('Contribution function was not saved for this line!')

    if hasattr(self, 'prepared'):
        if not self.prepared: self.prepare()

    name = 'cntrbf_{:06}'.format(self.kr + 1)
    mem = self.run.read_patch_save(name, fdim=0, lazy=True)[-1]

    if df is not None:
        f = self.i0 - self.ib + df
        mem = mem[f:f + 1]
    elif fslice is not None:
        mem = mem[fslice]

    if LTE:
        cntrbf = mem[..., 2:4, :]
    else:
        cntrbf = mem[..., 0:2, :]

    nf, nx, ny, nz, nset, nang = cntrbf.shape
    ltau = self.run.ltau
    get_mean = mean and self.run.dim == '3D'

    if get_mean:
        new_ltau = np.mean(ltau, axis=(0, 1))[::-1]

    all_angs = np.arange(self.run.n_ang)
    if ang is None and imu is None:
        angs = all_angs
    elif imu is not None:
        if imu < 0: imu = self.run.imus.max() + imu + 1
        angs = all_angs[self.run.imus == imu]
    elif ang is not None:
        angs = np.array([ang])

    wts = self.run.wts[angs]
    wts = wts / np.sum(wts)

    conversion = np.log(10) * self.run.tau / self.run.chi

    first_time = True
    for ang, w in zip(angs, wts):
        cang = np.array(cntrbf[..., ang])
        cf, net_chi = np.moveaxis(cang, -1, 0)
        vec = self.run.vec[ang]
        zz = self.run.zz / vec[-1] * 1e8
        tau = compute_tau_scale(net_chi, zz, axis=-1)
        cf = conversion * cf * np.exp(-tau)
        if self.run.dim == '3D':
            vec = vec * np.array([-1, -1, 1])
            cf = rotate_array(self.run.xx, self.run.yy, self.run.zz, cf, vec)

        if get_mean:
            yy = np.array(cf[..., ::-1]).copy()
            xx = ltau[..., ::-1].copy()
            newcf = interp_tau(new_ltau, xx, yy)
            cf = np.mean(newcf, axis=(1, 2))

        if first_time:
            cfunc = np.zeros_like(cf)
            first_time = False

        cfunc = cfunc + w * cf

    if get_mean: ltau = new_ltau
    if norm:
        normalisation = abs(np.trapz(cfunc, x=ltau))
        cfunc = cfunc / normalisation
    return ltau, cfunc.squeeze()


# =====================================================================
def plot_cntrbf(self, df=0, imu=None, ang=None, LTE=False, norm=False, mean=True):
    # -----------------------------------------------------------------
    ltau, cf = get_cntrbf(self, df=df, imu=imu, ang=ang, LTE=LTE, mean=mean, norm=norm)

    return ltau, cf


def get_synthetic_spectra(m3dis, run_path, min_x, max_x):
    run = m3dis.read(run_path)
    xx_nlte, yy_nlte = get_xx_yy(run, min_x, max_x, norm=True, LTE=False)
    xx_lte, yy_lte = get_xx_yy(run, min_x, max_x, norm=True, LTE=True)
    ltau_lte, cf_lte = plot_cntrbf(run.line[0], LTE=True)
    ltau_nlte, cf_nlte = plot_cntrbf(run.line[0], LTE=False)
    return xx_lte, yy_lte, xx_nlte, yy_nlte, ltau_lte, cf_lte, ltau_nlte, cf_nlte