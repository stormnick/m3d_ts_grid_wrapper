import os
import shutil
import numpy as np
from atom_package import write_atom_noReFormatting
from atmos_package import *
from combine_grids import addRec_to_NLTEbin
from m1d_output import m1d
import datetime
from dask.distributed import get_worker
import pickle


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)



def assign_temporary_directory(setup):
    worker = get_worker()
    try:
        _ = worker.temporary_directory
    except AttributeError:
        worker.temporary_directory = f"{setup.common_wd}/job_{np.random.random()}/"
        setup_temp_dirs(setup, worker.temporary_directory)
    try:
        _ = worker.atom_body
    except AttributeError:
        with open(os.path.join(setup.common_wd, "pickled_atom"), 'rb') as atom_pickled_body:
            worker.atom_body = pickle.load(atom_pickled_body)


def setup_temp_dirs(setup, temporary_directory):
    """
    Setting up and running an individual serial job of NLTE calculations
    for a set of model atmospheres (stored in setup.jobs[k]['atmos'])
    Several indidvidual jobs can be run in parallel, set ncpu in the config. file
    to the desired number of processes
    Note: Multi 1D expects all input files to be named only in upper case

    input:
    # TODO:
    (string) directory: common working directory, default: "./"
    (integer) k: an ID of an individual job within the run
    (object) setup: object of class setup, regulates a setup for the whole run
    """

    """
    Make sure that only one process at a time can access input files
    """
    # lock = multiprocessing.Lock()
    # lock.acquire()

    """ Make a temporary directory """
    mkdir(temporary_directory)

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'absdat']:
        os.symlink(os.path.join(setup.m1d_input, file), os.path.join(temporary_directory, file.upper()))

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink(setup.m1d_input_file, os.path.join(temporary_directory, 'INPUT'))

    """ Link executable """
    os.symlink(setup.m1d_exe, os.path.join(temporary_directory, 'mul23lus.x'))

    """
    What kind of output from M1D should be saved?
    Read from the config file, passed here through the object setup
    """
    #job.output.update({'write_ew': setup.write_ew, 'write_ts': setup.write_ts})
    """ Save EWs """
    if setup.write_ew == 1 or setup.write_ew == 2:
        # create file to dump output
        #job.output.update({'file_ew': temporary_directory + '/output_EW.dat'})
        with open(os.path.join(temporary_directory, 'output_EW.dat'), 'w') as f:
            f.write(
                "# Teff [K], log(g) [cgs], [Fe/H], A(X), stat. weight g_i, energy en_i, wavelength air [AA], osc. strength, EW(NLTE) [AA], EW(LTE) [AA], Vturb [km/s]    \n")

    elif setup.write_ew == 0:
        pass
    else:
        print("write_ew flag unrecognised, stoppped")
        exit(1)

    """ Output for TS? """
    if setup.write_ts == 1:
        # create a file to dump output from this serial job
        # array rec_len stores a length of the record in bytes
        #job.output.update({'file_4ts': temporary_directory + '/output_4TS.bin', \
        #                   'file_4ts_aux': temporary_directory + '/auxdata_4TS.txt', \
        #                   'rec_len': np.zeros(len(job.atmos), dtype=int)})
        # create the files
        with open(os.path.join(temporary_directory, 'output_4TS.bin'), 'wb') as f:
            pass
        with open(os.path.join(temporary_directory, 'auxdata_4TS.txt'), 'w') as f:
            f.write("# atmos ID, Teff [K], log(g) [cgs], [Fe/H], [alpha/Fe], mass, Vturb [km/s], A(X), pointer \n")
    elif setup.write_ts == 0:
        pass
    else:
        print("write_ts flag unrecognised, stopped")
        exit(1)
    #job.output.update({'save_idl1': setup.save_idl1})
    #if setup.save_idl1 == 1:
    #    job.output.update({'idl1_folder': setup.idl1_folder})

    # lock.release()
    #return job

def setup_multi_job(setup, job, temporary_directory):
    """
    Setting up and running an individual serial job of NLTE calculations
    for a set of model atmospheres (stored in setup.jobs[k]['atmos'])
    Several indidvidual jobs can be run in parallel, set ncpu in the config. file
    to the desired number of processes
    Note: Multi 1D expects all input files to be named only in upper case

    input:
    # TODO:
    (string) directory: common working directory, default: "./"
    (integer) k: an ID of an individual job within the run
    (object) setup: object of class setup, regulates a setup for the whole run
    """

    """
    Make sure that only one process at a time can access input files
    """
    #lock = multiprocessing.Lock()
    #lock.acquire()

    #mkdir(temporary_directory)

    job.output.update({'write_ew': setup.write_ew, 'write_ts': setup.write_ts})
    """ Save EWs """
    if job.output['write_ew'] == 1 or job.output['write_ew'] == 2:
        # create file to dump output
        job.output.update({'file_ew': temporary_directory + '/output_EW.dat'})
    else:
        job.output['file_ew'] = None

    """ Output for TS? """
    if job.output['write_ts'] == 1:
        # create a file to dump output from this serial job
        # array rec_len stores a length of the record in bytes
        job.output.update({'file_4ts': temporary_directory + '/output_4TS.bin', \
                           'file_4ts_aux': temporary_directory + '/auxdata_4TS.txt', \
                           'rec_len': np.zeros(len(job.atmo), dtype=int)})

    job.output.update({'save_idl1': setup.save_idl1})
    if setup.save_idl1 == 1:
        job.output.update({'idl1_folder': setup.idl1_folder})

    #lock.release()
    return job


def run_multi(job, atom, atmos, temporary_directory, common_wd, atom_body):
    """
    Run MULTI1D
    input:
    (object) setup:
    (object) job:
    (integer) i: index pointing to the current model atmosphere and abundance
                 (in job.atmos, job.abund)
                 this model atmosphere and abund will be used to run M1D
    (object) atom:  object of class model_atom
    """

    """ Create ATOM input file for M1D """
    write_atom_noReFormatting(atom, temporary_directory + '/ATOM', atom_body)

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, temporary_directory + '/ATMOS')
    write_dscale_m1d(atmos, temporary_directory + '/DSCALE')

    """ Go to directory and run MULTI 1D """
    os.chdir(temporary_directory)

    run_multi_exe()

    """ Read M1D output if M1D run was successful """
    if os.path.isfile('./IDL1'):
        out = m1d('./IDL1')

        """ print MULTI1D output to the temporary grid of EWs """
        if job.output['write_ew'] > 0:
            if job.output['write_ew'] == 1:
                mask = np.arange(out.nline)
            elif job.output['write_ew'] == 2:
                mask = np.where(out.nq[:out.nline] == max(out.nq[:out.nline]))[0]

            with open(job.output['file_ew'], 'a') as f:
                for kr in mask:
                    line = out.line[kr]
                    f.write("%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %20.10f %20.10f %20.4f # '%s'\n" \
                            % (atmos.teff, atmos.logg, atmos.feh, out.abnd, out.g[out.irad[kr]], out.ev[out.irad[kr]], \
                               line.lam0, out.f[kr], out.weq[kr], out.weqlte[kr], np.mean(atmos.vturb), atmos.id))

        """ save  MULTI1D output in a common binary file in the format for TS """
        if job.output['write_ts'] == 1:
            faux = open(job.output['file_4ts_aux'], 'a')
            # append record to binary grid file
            with np.errstate(divide='ignore'):
                depart = np.array((out.n / out.nstar), dtype='f8')
                # transpose to match Fortran order of things
                # (nk, ndep)
                depart = depart.T
            record_len = addRec_to_NLTEbin(job.output['file_4ts'], atmos.id, out.ndep, out.nk, out.tau, depart)

            faux.write(" '%s' %10.4f %10.4f %10.4f %10.4f %10.2f %10.2f %10.4f %60.0f \n" \
                       % (atmos.id, atmos.teff, atmos.logg, atmos.feh, atmos.alpha, atmos.mass, np.mean(atmos.vturb),
                          out.abnd, record_len))
            faux.close()

        if job.output['save_idl1'] == 0:
            os.remove('./IDL1')
        elif job.output['save_idl1'] == 1:
            destin = job.output['idl1_folder'] + "/idl1.%s_%s_A(X)_%5.5f" % (
            atmos.id, atom.element.replace(' ', ''), atom.abund)
            shutil.move('./IDL1', destin)
    # no IDL1 file created after the run
    else:
        print("IDL1 file not found for %s A(X)=%.2f" % (atmos.id, atom.abund))

    out = None
    os.chdir(common_wd)


def run_multi_exe():
    os.system('time ./mul23lus.x')


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


def write_atmo_abundance(atmo: ModelAtmosphere, elemental_abundance_m1d: dict, new_abund_file_locaiton: str, atom_abund: float, atom_element: str):
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


def run_serial_job(job, temporary_directory):
    job = setup_multi_job(job, temporary_directory)


    # model atom is only read once
    atom = setup.atom
    atmos = ModelAtmosphere()
    atmos.read(file=job.atmo, file_format=setup.atmos_format)
    # scale abundance with [Fe/H] of the model atmosphere
    if np.isnan(atmos.feh):
        atmos.feh = 0.0
    if not atom.element.lower() == 'h':
        atom.abund = job.abund + atmos.feh

    write_atmo_abundance(atmos, setup.elemental_abundance_m1d, os.path.join(temporary_directory, "ABUND"), atom.abund, atom.element)
    run_multi(job, atom, atmos, temporary_directory, setup.common_wd, worker.atom_body)

    job_return_info = [job.output['file_ew'], job.output['file_4ts'], job.output['file_4ts_aux']]   #{'file_ew': , 'file_4ts', 'file_4ts_aux'}
    job = None
    # shutil.rmtree(job['tmp_wd'])
    return job_return_info
