from __future__ import annotations

# Created by storm at 17.10.24

import logging
import shutil
import subprocess
import numpy as np
import os
import tempfile
import marcs_class


def run_m3dis(mpi_cores, temp_path, input_in, stderr, stdout):
    # Write the input data to a temporary file
    # TODO: check this solution because temp direction might mess up something
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(bytes(input_in, "utf-8"))
        temp_file_name = temp.name

    #print(input_in)

    # check if file exists
    #if not os.path.exists("./dispatch.x"):
    #    print("File does not exist")
    #else:
    #    print("File exists")

    # Create a copy of the current environment variables
    env = os.environ.copy()
    # Set OMP_NUM_THREADS for the subprocess

    env['OMP_NUM_THREADS'] = str(mpi_cores)

    # Now, you can use temp_file_name as an argument to dispatch.x
    pr1 = subprocess.Popen(
        [
            "./dispatch.x",
            temp_file_name,
        ],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )
    # pr1.stdin.write(bytes(input_in, "utf-8"))
    stdout_bytes, stderr_bytes = pr1.communicate()

    # Don't forget to remove the temporary file at some point
    os.unlink(temp_file_name)
    return pr1, stderr_bytes


def call_m3dis(m3dis_path, temp_path, atmo_model_path, atom_path, atom_abund, convlim, iterations_max, use_absmet, absmet_file_path, hash_table_size, use_precomputed_depart=False, verbose=False, precomputed_depart_path=None):
    atmo_param = f"atmos_format='Marcs'"
    if use_precomputed_depart and precomputed_depart_path is not None:
        precomputed_depart = f"precomputed_depart='{precomputed_depart_path}'"
    else:
        precomputed_depart = ""
    atom_params = (f"&atom_params        atom_file='{atom_path}' "
                   f"convlim={convlim} use_atom_abnd=F exclude_trace_cont=T exclude_from_line_list=T "
                   f"{precomputed_depart} abundance={atom_abund:.2f}/\n")

    if use_absmet:
        absmet_file = absmet_file_path
    else:
        absmet_file = ""

    # 0.010018 0.052035 0.124619 0.222841 0.340008 0.468138 0.598497 0.722203 0.830825 0.916958 0.974726 1.000000
    # turbospectrum angles
    output = {}
    config_m3dis = (f"! -- Parameters defining the run -----------------------------------------------\n\
&io_params          datadir='{temp_path}' gb_step=100.0 do_trace=F /\n\
&timer_params       sec_per_report=1e8 /\n\
&atmos_params       dims=1 save_atmos=T atmos_file='{atmo_model_path}' {atmo_param}/\n{atom_params}\
&m3d_params         decouple_continuum=T verbose=2 n_nu=1 maxiter={iterations_max}/\n\
&spectrum_params    daa=0.1 aa_blue=5000 aa_red=5001 /\n\
&composition_params isotope_file='{temp_path}/../../isotopes' abund_file='{temp_path}/../../abund_to_use' {absmet_file}/\n\
&task_list_params   hash_table_size={hash_table_size} /\n")

    if verbose:
        print(config_m3dis)

    if verbose:
        stdout = None
        stderr = subprocess.STDOUT
    else:
        stdout = open("/dev/null", "w")
        stderr = subprocess.STDOUT

    cwd = os.getcwd()

    try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
        os.chdir(os.path.join(m3dis_path, ""))  #
        #print(os.getcwd())
        pr1, stderr_bytes = run_m3dis(1, temp_path, config_m3dis, stderr, stdout)
    except subprocess.CalledProcessError:
        output["errors"] = "babsma failed with CalledProcessError"
        return output
    finally:
        os.chdir(cwd)
    if stderr_bytes is None:
        stderr_bytes = b""
    if pr1.returncode != 0:
        output["errors"] = f"m3dis failed with return code {pr1.returncode} {stderr_bytes.decode('utf-8')}"

    return output

def synthesize_spectra(m3dis_path, temp_path, atmo_model_path, atom_path, atom_abund, convlim, iterations_max, use_absmet, absmet_file_path, hash_table_size, use_precomputed_depart, verbose, precomputed_depart_path=None):
    try:
        # clean temp directory
        save_file_dir = os.path.join(temp_path, "save")
        if os.path.exists(save_file_dir):
            # just in case it fails, so that it doesn't reuse the old files
            shutil.rmtree(save_file_dir)
        logging.debug("Running m3dis")
        output = call_m3dis(m3dis_path, temp_path, atmo_model_path, atom_path, atom_abund, convlim, iterations_max, use_absmet, absmet_file_path, hash_table_size, use_precomputed_depart=use_precomputed_depart, verbose=verbose, precomputed_depart_path=precomputed_depart_path)
        if "errors" in output:
            print(output["errors"], "m3dis failed")
            return False
    except (FileNotFoundError, ValueError, TypeError) as error:
        print(f"Interpolation failed? {error}")
        return False
    return True
