import os
import shutil
import numpy as np
import datetime
from dask.distributed import get_worker
import pickle
from run_m3dis import synthesize_spectra
from convert_grid_to_ts import *


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)

def collect_output(setup, jobs, jobs_amount):
    today = datetime.date.today().strftime("%b-%d-%Y")

    written_comment_ew = False
    written_comment_aux = False

    done_ew_file_names = set()
    done_ts_file_names = set()

    # order of the job:
    # job = [job.output['file_ew'], job.output['file_4ts'], job.output['file_4ts_aux']]

    """ Collect all EW grids into one """
    datetime0 = datetime.datetime.now()
    if setup.write_ew > 0:
        print("Collecting grids of EWs")
        with open(os.path.join(setup.common_wd, 'output_EWgrid_%s.dat' % (today)), 'w') as com_f:
            for job in jobs:
                ew_file = job[0]
                if ew_file not in done_ew_file_names:
                    data = open(ew_file, 'r').readlines()
                    com_f.writelines(data)
                    done_ew_file_names.add(ew_file)
        """ Checks to raise warnings if there're repeating entrances """
        with open(os.path.join(setup.common_wd, 'output_EWgrid_%s.dat' % (today)), 'r') as f:
            data_all = f.readlines()
        params = []
        for line in data_all:
            if not line.startswith('#'):
                abund = line.split()[3]
                atmosID = line.split()[-1]
                wave = line.split()[6]
                if not [wave, abund, atmosID] in params:
                    params.append([wave, abund, atmosID])
                else:
                    print("WARNING: found repeating entrance at \n %s AA  A(X)=%s, atmos: %s " \
                          % (wave, abund, atmosID))
        datetime1 = datetime.datetime.now()
        print(datetime1 - datetime0)
        print(10 * "-")
        """ #TODO sort the grids of EWs """

    """ Collect all TS formatted NLTE grids into one """
    datetime0 = datetime.datetime.now()
    if setup.write_ts > 0:
        print("Collecting TS formatted NLTE grids")
        com_f = open(os.path.join(setup.common_wd, 'output_NLTEgrid4TS_%s.bin' % (today)), 'wb')
        com_aux = open(os.path.join(setup.common_wd, 'auxData_NLTEgrid4TS_%s.dat' % (today)), 'w')

        header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n" + \
                 "NLTE element: %s \n" % (setup.atom.element) + \
                 "Model atom: %s \n" % (setup.atom_id) + \
                 "Comments: '%s' \n" % (setup.atom.info) + \
                 "Number of records: %10.0f \n" % (jobs_amount) + \
                 f"Computed with MULTI 1D (using EkaterinaSe/wrapper_multi (github)), {today} \n"
        header = str.encode('%1000s' % (header))
        com_f.write(header)

        # Fortran starts with 1 while Python starts with 0
        pointer = len(header) + 1

        for job in jobs:
            ts_bin_file = job[1]
            if ts_bin_file not in done_ts_file_names:
                # departure coefficients in binary format
                done_ts_file_names.add(ts_bin_file)
                with open(ts_bin_file, 'rb') as f:
                    com_f.write(f.read())
                for line in open(job[2], 'r').readlines():
                    if not line.startswith('#'):
                        rec_len = int(line.split()[-1])
                        com_aux.write('    '.join(line.split()[0:-1]))
                        com_aux.write("%20.0f \n" % (pointer))
                        pointer = pointer + rec_len
                    # simply copy comment lines
                    else:
                        if not written_comment_aux:
                            com_aux.write(line)
                            written_comment_aux = True
                        else:
                            pass

        com_f.close()
        com_aux.close()
        datetime1 = datetime.datetime.now()
        print(datetime1 - datetime0)
        print(10 * "-")

def write_atmo_abundance(atmo, elemental_abundance_m1d: dict, new_abund_file_locaiton: str, atom_abund: float, atom_element: str):
    """
    Scales abundance according to the atmosphere. Either takes already atmospheric abundance or scaled the one in M1D
    according to metallicity (except H and He). Uses only elements that were already written in the M1D ABUND file
    """
    with open(new_abund_file_locaiton, "w") as new_file_to_write:
        for element in elemental_abundance_m1d:
            if atmo.atmospheric_abundance is not None:
                element_name = element.upper()
                elemental_abundance = atmo.atmospheric_abundance[element.upper()]
            else:
                element_name = element.upper()
                elemental_abundance = elemental_abundance_m1d[element]
                if element_name != "H" and element_name != "HE":
                    elemental_abundance += atmo.feh
            if atom_element == "CH" and element_name == "C":
                elemental_abundance = atom_abund
            if atom_element == "CN" and element_name == "N":
                elemental_abundance = atom_abund
            if atom_element.upper() == element_name:
                elemental_abundance = atom_abund
            new_file_to_write.write(f"{element_name:<4}{elemental_abundance:5,.2f}\n")

def choose_absmet_file(absmet_file_global_path, metallicity):
    # check if feh is almost 0
    #absmet_file_global_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/absmet/"
    #absmet_file_global_path = "/Users/storm/PhD_2022-2025/m3dis_useful_stuff/absmet_files/"
    if np.abs(metallicity) < 0.01:
        # /mnt/beegfs/gemini/groups/bergemann/users/storm/data/absmet/OPACITIES/M+0.00a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00
        absmet_file = f"OPACITIES/M+0.00a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00/metals_noMnCrCoNi.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
    # check if feh is almost -1 or -0.5
    elif np.abs(metallicity + 1) < 0.01 or np.abs(metallicity + 0.5) < 0.01:
        absmet_file = f"m-1.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=T"
    # check if feh is almost -2
    elif np.abs(metallicity + 2) < 0.01:
        absmet_file = f"OPACITIES/M-2.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
    # check if feh is almost -3
    elif np.abs(metallicity + 3) < 0.01:
        absmet_file = f"OPACITIES/M-2.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
    # check if feh is almost -4
    elif np.abs(metallicity + 4) < 0.01:
        absmet_file = f"OPACITIES/M-4.00a+0.40c+0.00n+0.00o+0.40r+0.00s+0.00/metals_noMnCrCoNi.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}' absmet_big_end=F"
    # check if feh is almost +0.5
    elif np.abs(metallicity - 0.5) < 0.01:
        absmet_file = f"OPACITIES/M+0.50a+0.00c+0.00n+0.00o+0.00r+0.00s+0.00/metals_noMnCrCoNi.x01"
        absmet_file = f"absmet_file='{absmet_file_global_path}/{absmet_file}'  absmet_big_end=F"
    #    &composition_params absmet_file='/u/nisto/data/absmet//m1/metals.x01' absmet_big_end=T /
    # absmet_file = f"absmet_file='{os.path.join(self.departure_file_path, '')}' absmet_big_end=T"
    return absmet_file

def run_serial_job(atom_abund, atmo, atmos_path, temporary_directory, m3dis_path, convlim, iterations_max, use_absmet, absmet_global_path, hash_table_size, use_precomputed_depart=False, verbose=True):
    #job = setup_multi_job(job, temporary_directory)
    atmo_path = os.path.join(atmos_path, atmo)

    atmo_name = os.path.basename(atmo)
    teff, logg, mass, vmic, feh, alpha, c, n, o, r, s = extract_atmo_info_name(atmo_name)

    atom_path = os.path.join(temporary_directory, "atom.my_atom")
    # random number
    random_number = str(np.random.random())
    m3d_path_run = os.path.join(temporary_directory, "temp_directory_for_m3d", random_number)
    os.makedirs(m3d_path_run)
    temporary_grid_save_aux_file = os.path.join(temporary_directory, "temporary_grid", f"{random_number}.txt")
    temporary_grid_save_bin_file = os.path.join(temporary_directory, "temporary_grid", f"{random_number}.bin")

    if use_absmet:
        absmet_file_path = choose_absmet_file(absmet_global_path, feh)
    else:
        absmet_file_path = ""

    completed_bool = synthesize_spectra(m3dis_path, m3d_path_run, atmo_path, atom_path, atom_abund, convlim, iterations_max, use_absmet, absmet_file_path, hash_table_size, use_precomputed_depart, verbose)

    if completed_bool:
        marcs_model = MARCSModel(atmo_path)
        log_tau, depart_values_nlte, atom_levels = read_m3dis_bin(os.path.join(m3d_path_run, "save", ""))
        depart_values_nlte_interp = get_marcs_depart_interpolated(
            log_tau, depart_values_nlte, marcs_model.lgTau5, atom_levels
        )
        log_tau_500 = marcs_model.lgTau5
        record_len = add_record_to_binary_file(
            temporary_grid_save_bin_file,
            atmo_name,
            log_tau_500,
            depart_values_nlte_interp,
        )
        add_record_to_aux_file(
            temporary_grid_save_aux_file,
            atmo_name,
            atom_abund,
            record_len,
        )
    else:
        record_len = 0

    # clean m3d_path_run
    shutil.rmtree(m3d_path_run)

    return record_len
