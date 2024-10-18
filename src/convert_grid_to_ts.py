import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import perf_counter
from marcs_class import MARCSModel
import re

matplotlib.use("MacOSX")

# Created by storm at 16.10.24


def compute_tau_scale(chi, zz, axis=-1):
    # -----------------------------------------------------------------
    chi = np.moveaxis(chi, axis, 0)
    chi = chi[::-1]
    dz = np.diff(zz)[::-1]
    dchi = np.array(np.zeros_like(chi))
    tau1 = 0.5 * (chi[1] + chi[0]) * dz[0]
    tau2 = 0.5 * (chi[2] + chi[1]) * dz[1] + tau1
    mask = tau2 > 0
    dchi[0, mask] = (tau1 * tau1)[mask] / tau2[mask]
    dchi[1:] = (0.5 * (chi[1:] + chi[:-1]).T * dz).T
    tau = np.cumsum(dchi, axis=0)
    tau = tau[::-1]
    tau = np.moveaxis(tau, 0, axis)
    return tau


def read_m3dis_bin(
    sfolder, file="atom_patch_001000.bin", dtype="<f4"
):
    # dims = (1, 1, dims_atmo, dims_atom_levels, 2, 1)
    # depart_values = np.memmap(
    #    sfolder + file, dtype=dtype, mode="r+", shape=dims, order="F"
    # )
    # depart_values = np.squeeze(depart_values)

    with open(sfolder + "atom_patch_meta.txt", "r") as mesh:
        mesh.readline()
        _, _, _, _, _, dims_atom_levels, _, _ = [i for i in mesh.readline().split()]
        dims_atom_levels = int(dims_atom_levels)

    with open(sfolder + "../atmos_mesh.txt", "r") as mesh:
        nx, ny, nz = [int(i) for i in mesh.readline().split()]
        dx, dy, dz = [float(i) for i in mesh.readline().split()]
    zz = np.arange(nz) * dz

    dims_atmo = nz

    with open(sfolder + file, "rb") as fbin:
        depart_values = np.fromfile(
            fbin, dtype=dtype, count=dims_atmo * dims_atom_levels * 2
        ).reshape(dims_atmo, dims_atom_levels, 2, order="F")

    with open(sfolder + "atmos_001000.bin", "rb") as atmos_file:
        # 32         1         1       256         7         1
        tau = np.fromfile(atmos_file, dtype=dtype).reshape(dims_atmo, 7, order="F")[
            :, -1
        ]

        tau = compute_tau_scale(tau, zz * 1e8, axis=-1)
        log_tau = np.log10(tau)
        # print all the data in the file

    return log_tau, depart_values[:, :, 1], dims_atom_levels


def get_marcs_depart_interpolated(log_tau, depart_values_nlte, marcs_model_lgTau5, atom_levels):
    # need to interpolate log_tau to marcs_model.lgTau5
    depart_values_nlte_interp = np.zeros((marcs_model_lgTau5.size, atom_levels))

    # if log_tau is not in increasing order, reverse it
    if log_tau[0] > log_tau[-1]:
        log_tau = log_tau[::-1]
        depart_values_nlte = depart_values_nlte[::-1]

    for i in range(atom_levels):
        # Perform the interpolation
        depart_values_interpolated = np.interp(
            marcs_model_lgTau5, log_tau, depart_values_nlte[:, i]
        )

        depart_values_nlte_interp[:, i] = depart_values_interpolated

    return depart_values_nlte_interp


def add_record_to_binary_file(bin_file, atmos_name, logtau, depart_values):
    record_len = 0

    # use append binary
    with open(bin_file, "ab") as fbin:
        record_len = record_len + 500
        # str.encode('%500s' % atmosID) length 500
        fbin.write(str.encode("%500s" % atmos_name.replace(".mod", "")))

        ndep = len(logtau)
        record_len = record_len + 4
        fbin.write(int(ndep).to_bytes(4, "little"))

        nk = len(depart_values[0])
        record_len = record_len + 4
        fbin.write(int(nk).to_bytes(4, "little"))

        fbin.write(np.array(np.power(10, logtau), dtype="f8").tobytes())
        record_len = record_len + ndep * 8
        fbin.write(np.array(depart_values.T, dtype="f8").tobytes())
        record_len = record_len + ndep * nk * 8

    return record_len


def add_record_to_aux_file(faux, atmos_name, abundance, pointer):
    # model["temperature"], model["log_g"], model["mass"], model["turbulence"], model["metallicity"], model["a"], model["c"], model["n"], model["o"], model["r"], model["s"]
    teff, logg, mass, vmic, feh, alpha, c, n, o, r, s = extract_atmo_info_name(
        atmos_name
    )
    with open(faux, "a") as faux:
        faux.write(
            " '%s' %10.4f %10.4f %10.4f %10.4f %10.2f %10.2f %10.4f %60.0f \n"
            % (
                atmos_name.replace(".mod", ""),
                teff,
                logg,
                feh,
                alpha,
                mass,
                vmic,
                abundance,
                pointer,
            )
        )


def create_new_bin_aux_files(aux_file, bin_file, description_bin):
    today = datetime.date.today().strftime("%b-%d-%Y")
    with open(bin_file, "wb") as fbin:
        header = (
            "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n"
            + f"{description_bin} \n"
            + f"Computed with DISPATCH@MULTI3D, {today} \n"
        )
        header = str.encode("%1000s" % header)
        fbin.write(header)

    with open(aux_file, "w") as faux:
        header = "# atmos ID, Teff [K], log(g) [cgs], [Fe/H], [alpha/Fe], mass, Vturb [km/s], A(X), pointer\n"
        faux.write(header)

    return 1000 + 1


def extract_atmo_info_name(marcs_name):
    pattern = (
        r"([sp])(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_"
        r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....).mod"
    )
    re_test = re.match(pattern, marcs_name)
    assert re_test is not None, "Could not parse MARCS model filename <{}>".format(
        marcs_name
    )
    try:
        model = {
            "spherical": re_test.group(1),
            "temperature": float(re_test.group(2)),
            "log_g": float(re_test.group(3)),
            "mass": float(re_test.group(4)),
            "turbulence": float(
                re_test.group(5)
            ),  # micro turbulence assumed in MARCS atmosphere, km/s
            "model_type": re_test.group(6),
            "metallicity": float(re_test.group(7)),
            "a": float(re_test.group(8)),
            "c": float(re_test.group(9)),
            "n": float(re_test.group(10)),
            "o": float(re_test.group(11)),
            "r": float(re_test.group(12)),
            "s": float(re_test.group(13)),
        }
    except ValueError:
        # logging.info("Could not parse MARCS model filename <{}>".format(filename))
        raise ValueError("Could not parse MARCS model filename <{}>".format(marcs_name))
    return (
        model["temperature"],
        model["log_g"],
        model["mass"],
        model["turbulence"],
        model["metallicity"],
        model["a"],
        model["c"],
        model["n"],
        model["o"],
        model["r"],
        model["s"],
    )


if __name__ == '__main__':
    sfolder = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data_test/input_ba_test_ba6//save/"
    # sfolder = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data_test/input_ba_test_ba6_precomp/save/"
    file = "atom_patch_001000.bin"
    dtype = "<f4"
    dims = (1, 1, 256, 6, 2, 1)
    marcs_model_name = (
        "p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    )
    abundance = 0.0

    marcs_model = MARCSModel(f"input_multi3d/atmos/{marcs_model_name}")
    atom_levels = 6

    log_tau, depart_values_nlte = read_m3dis_bin(sfolder, atom_levels)
    depart_values_nlte_interp = get_marcs_depart_interpolated(
        log_tau, depart_values_nlte, marcs_model.lgTau5
    )
    log_tau_500 = marcs_model.lgTau5

    # create new binary and aux files
    bin_file = "test_output_NLTEgrid4TS_combined.bin"
    aux_file = "test_auxData_NLTEgrid4TS_combined.dat"
    description_bin = "Test"
    pointer = create_new_bin_aux_files(aux_file, bin_file, description_bin)
    record_len = add_record_to_binary_file(
        bin_file,
        marcs_model_name,
        log_tau_500,
        depart_values_nlte_interp,
    )
    add_record_to_aux_file(
        aux_file,
        marcs_model_name,
        abundance,
        pointer,
    )
    pointer = pointer + record_len
