from __future__ import annotations

# Created by storm at 17.10.24

import logging
import shutil
import subprocess
import numpy as np
import os
import tempfile
import marcs_class
from solar_abundances import periodic_table, solar_abundances
from solar_isotopes import solar_isotopes


class M3disCall:
    def __init__(self, m3dis_path: str, model_atom_path: str,
                hash_table_size=None, mpi_cores=None, iterations_max=None, convlim=None):
        self.m3dis_path = m3dis_path
        self.mpi_cores: int = mpi_cores
        self.n_nu = 1
        self.hash_table_size = hash_table_size
        self.iterations_max = iterations_max
        self.convlim = convlim
        self.dims = 1
        self.skip_linelist = True
        self.lambda_min = 5000
        self.lambda_max = 5001
        self.lambda_delta = 0.5
        self.model_atom_path = model_atom_path
        self.use_marcs_directly = True
        self.metallicity = 0.0

    def configure(self, free_abundances=None, verbose=None, temp_directory=None, nlte_flag: bool = None,
                  atmosphere_dimension=None, model_atom_file=None):
        if free_abundances is not None:
            self.free_abundances = free_abundances  # [X/H]
        if verbose is not None:
            self.verbose = verbose
        if temp_directory is not None:
            self.tmp_dir = temp_directory
        if nlte_flag is not None:
            self.nlte_flag = nlte_flag
        if atmosphere_dimension is not None:
            self.atmosphere_dimension = atmosphere_dimension
        if model_atom_file is not None:
            self.model_atom_file = model_atom_file

    def run_m3dis(self, input_in, stderr, stdout):
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

        env['OMP_NUM_THREADS'] = str(self.mpi_cores)

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

    def write_abund_file(self):
        # file path is temp_dir + abund
        file_path = os.path.join(self.tmp_dir, "abund_to_use")
        # open file
        with open(file_path, "w") as file:
            # write the number of elements
            # write the elements and their abundances
            for element in periodic_table:
                if element != "":
                    if element == "H" or element == "He":
                        abundance_to_write = solar_abundances[element]
                    else:
                        if element in self.free_abundances:
                            # Here abundance is passed as [X/H], so we need to add the solar abundance to convert to A(X)
                            # A(X)_star = A(X)_solar + [X/H]
                            abundance_to_write = self.free_abundances[element] + solar_abundances[element]
                        else:
                            # If the element is not in the free abundances, we assume it has the solar scaled abundance
                            # A(X)_star = A(X)_solar + [Fe/H]
                            abundance_to_write = solar_abundances[element] + self.metallicity
                        if self.use_marcs_directly:
                            # if 3D, we need to subtract the metallicity from the abundance, because it auto scales (adds it) in M3D with FeH already
                            abundance_to_write = abundance_to_write - self.metallicity
                    file.write(f"{element:<4} {abundance_to_write:>6.3f}\n")
                    logging.debug(f"{element:<4} {abundance_to_write:>6.3f}")
        return file_path



    def call_m3dis(self, use_precomputed_depart=False):
        abund_file_path = self.write_abund_file()
        isotope_file_path = self.write_isotope_file()

        atmo_param = f"atmos_format='Marcs"
        atmos_path = os.path.join(self.marcs_grid_path, self.marcs_model_name)

        atom_path = self.model_atom_path
        atom_files = list(self.model_atom_file.keys())
        atom_file_element = atom_files[0]
        if len(atom_files) > 1:
            print(f"Only one atom file is allowed for NLTE: m3dis, using the first one {atom_file_element}")
        if use_precomputed_depart:
            precomputed_depart = f"precomputed_depart='{os.path.join(self.tmp_dir, '../precomputed_depart', '')}'"
        else:
            precomputed_depart = ""
        atom_params = (f"&atom_params        atom_file='{os.path.join(atom_path, self.model_atom_file[atom_file_element])}' "
                       f"convlim={self.convlim} use_atom_abnd=F exclude_trace_cont=T abund= exclude_from_line_list=T "
                       f"{precomputed_depart}/\n")

        linelist_parameters = ""

        if False:
            # check if feh is almost 0
            absmet_file_global_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/absmet/"
            absmet_file_global_path = "/Users/storm/PhD_2022-2025/m3dis_useful_stuff/absmet_files/"
            if np.abs(self.metallicity) < 0.01:
                # /mnt/beegfs/gemini/groups/bergemann/users/storm/data/absmet/OPACITIES/M+0.00a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00
                absmet_file = f"OPACITIES/M+0.00a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00/metals_noMnCrCoNi.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
            # check if feh is almost -1 or -0.5
            elif np.abs(self.metallicity + 1) < 0.01 or np.abs(self.metallicity + 0.5) < 0.01:
                absmet_file = f"m-1.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=T"
            # check if feh is almost -2
            elif np.abs(self.metallicity + 2) < 0.01:
                absmet_file = f"OPACITIES/M-2.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
            # check if feh is almost -3
            elif np.abs(self.metallicity + 3) < 0.01:
                absmet_file = f"OPACITIES/M-2.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
            # check if feh is almost -4
            elif np.abs(self.metallicity + 4) < 0.01:
                absmet_file = f"OPACITIES/M-4.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
            # check if feh is almost +0.5
            elif np.abs(self.metallicity - 0.5) < 0.01:
                absmet_file = f"OPACITIES/M+0.50a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00/metals_noMnCrCoNi.x01"
                absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}'  absmet_big_end=F"
            #    &composition_params absmet_file='/u/nisto/data/absmet//m1/metals.x01' absmet_big_end=T /
            # absmet_file = f"absmet_file='{os.path.join(self.departure_file_path, '')}' absmet_big_end=T"
        else:
            absmet_file = ""

        # 0.010018 0.052035 0.124619 0.222841 0.340008 0.468138 0.598497 0.722203 0.830825 0.916958 0.974726 1.000000
        # turbospectrum angles
        output = {}
        config_m3dis = (f"! -- Parameters defining the run -----------------------------------------------\n\
&io_params          datadir='{self.tmp_dir}' gb_step=100.0 do_trace=F /\n\
&timer_params       sec_per_report=1e8 /\n\
&atmos_params       dims={self.dims} save_atmos=T atmos_file='{atmos_path}' {atmo_param}/\n{atom_params}\
&m3d_params         decouple_continuum=T verbose=2 n_nu={self.n_nu} maxiter={self.iterations_max} quad_scheme='set_a2' long_scheme='custom' custom_mu='0.010 0.052 0.124 0.223 0.340 0.468 0.598 0.722 0.831 0.917 0.975 1.000'/\n\
{linelist_parameters}\
&composition_params isotope_file='{isotope_file_path}' abund_file='{abund_file_path}' {absmet_file}/\n\
&task_list_params   hash_table_size={self.hash_table_size} /\n")

        if self.verbose:
            print(config_m3dis)

        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open("/dev/null", "w")
            stderr = subprocess.STDOUT

        cwd = os.getcwd()

        try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
            os.chdir(os.path.join(self.m3dis_path, ""))  #
            #print(os.getcwd())
            pr1, stderr_bytes = self.run_m3dis(config_m3dis, stderr, stdout)
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

        # Return output
        # output["return_code"] = pr.returncode
        # output["output_file"] = os_path.join(
        #    self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra)
        # )
        return output

    def synthesize_spectra(self, use_precomputed_depart=False):
        try:
            logging.debug("Running m3dis and atmosphere")
            logging.debug("Cleaning temp directory")
            # clean temp directory
            save_file_dir = os.path.join(self.tmp_dir, "save")
            if os.path.exists(save_file_dir):
                # just in case it fails, so that it doesn't reuse the old files
                shutil.rmtree(save_file_dir)
            logging.debug("Running m3dis")
            output = self.call_m3dis(use_precomputed_depart=use_precomputed_depart)
            if "errors" in output:
                print(output["errors"], "m3dis failed")
        except (FileNotFoundError, ValueError, TypeError) as error:
            print(f"Interpolation failed? {error}")

def write_isotope_file(isotope_dir):
    atomic_weights_path = "atomicweights.dat"
    # check if file exists
    if not os.path.exists(atomic_weights_path):
        # add ../ to the path
        atomic_weights_path = os.path.join("../", atomic_weights_path)
    m3d_isotopes_file_path = os.path.join(isotope_dir, "isotopes")
    # check if file exists
    if not os.path.exists(m3d_isotopes_file_path):
        free_isotopes = solar_isotopes
        elements_atomic_mass_number = free_isotopes.keys()

        # elements now consists of e.g. '3.006'. we want to convert 3
        elements_atomic_number = [int(float(element.split(".")[0])) for element in elements_atomic_mass_number]
        # count the number of each element, such that we have e.g. 3: 2, 4: 1, 5: 1
        elements_count = {element: elements_atomic_number.count(element) for element in elements_atomic_number}
        # remove duplicates
        elements_atomic_number_unique = set(elements_atomic_number)
        separator = "_"  # separator between sections in the file from NIST


        atomic_weights = {}
        with open(atomic_weights_path, "r") as file:
            skip_section = True
            current_element_atomic_number = 0
            for line in file:
                if line[0] != separator and skip_section:
                    continue
                elif line[0] == separator and skip_section:
                    skip_section = False
                    continue
                elif line[0] != separator and not skip_section and current_element_atomic_number == 0:
                    current_element_atomic_number_to_test = int(line.split()[0])
                    if current_element_atomic_number_to_test not in elements_atomic_number_unique:
                        skip_section = True
                        continue
                    current_element_atomic_number = current_element_atomic_number_to_test
                    atomic_weights[current_element_atomic_number] = {}
                    # remove any spaces and anything after (
                    atomic_weights[current_element_atomic_number][int(line[8:12].replace(" ", ""))] = \
                    line[13:32].replace(" ", "").split("(")[0]
                elif line[0] != separator and not skip_section and current_element_atomic_number != 0:
                    atomic_weights[current_element_atomic_number][int(line[8:12].replace(" ", ""))] = atomic_weight = \
                    line[13:32].replace(" ", "").split("(")[0]
                elif line[0] == separator and not skip_section and current_element_atomic_number != 0:
                    current_element_atomic_number = 0

        """
        format:
        Li    2
           6   6.0151   0.0759
           7   7.0160   0.9241
        """

        # open file

        with open(m3d_isotopes_file_path, "w") as file:
            # write element, then number of isotopes. next lines are isotope mass and abundance
            current_element_atomic_number = 0
            for element, isotope in free_isotopes.items():
                element_atomic_number = int(float(element.split(".")[0]))
                element_mass_number = int(float(element.split(".")[1]))
                if current_element_atomic_number != element_atomic_number:
                    # elements now consists of e.g. '3.006'. we want to convert 3 to and 6
                    current_element_atomic_number = element_atomic_number
                    file.write(
                        f"{periodic_table[element_atomic_number]:<5}{elements_count[element_atomic_number]:>2}\n")

                file.write(
                    f"{int(element_mass_number):>4} {float(atomic_weights[element_atomic_number][element_mass_number]):>12.8f} {isotope:>12.8f}\n")
    return m3d_isotopes_file_path


if __name__ == '__main__':
    teff, logg, feh, vmic = 5777, 4.44, 0.0, 1.0

    model_atmosphere_grid_path = "/Users/storm/docker_common_folder/TSFitPy/input_files/model_atmospheres/1D/"
    model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    temp_dir = "./temppp_dirrr/"
    os.makedirs(temp_dir, exist_ok=True)

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        model_atmosphere_list, M3disCall.marcs_parameters_to_ignore)

    m3dis_class = M3disCall(
        m3dis_path=None,
        interpol_path=None,
        line_list_paths=None,
        marcs_grid_path=model_atmosphere_grid_path,
        marcs_grid_list=model_atmosphere_list,
        model_atom_path=None,
        departure_file_path=None,
        aux_file_length_dict=None,
        model_temperatures=model_temperatures,
        model_logs=model_logs,
        model_mets=model_mets,
        marcs_value_keys=marcs_value_keys,
        marcs_models=marcs_models,
        marcs_values=marcs_values,
        m3dis_python_module=None,
        n_nu=None,
        hash_table_size=None,
        mpi_cores=None,
        iterations_max=None,
        convlim=None,
        snap=None,
        dims=None,
        nx=None,
        ny=None,
        nz=None
    )
    teff, logg, feh, vmic = 5777, 4.44, 0.0, 1.0

    teff = 4665
    logg = 1.64
    feh = -2.5
    vmic = 2.0

    fehs = [-2.5, -2.3, -2.1, -1.9, -2.7, -2.9, -3.1]

    for feh in fehs:
        m3dis_class.configure(t_eff=teff, log_g=logg, metallicity=feh, turbulent_velocity=vmic,
                                        lambda_delta=None, lambda_min=None, lambda_max=None,
                                        free_abundances=None, temp_directory=temp_dir, nlte_flag=False,
                                        verbose=None,
                                        atmosphere_dimension="1D", windows_flag=None,
                                        segment_file=None, line_mask_file=None,
                                        depart_bin_file=None, depart_aux_file=None,
                                        model_atom_file=None)
        m3dis_class.calculate_atmosphere()
