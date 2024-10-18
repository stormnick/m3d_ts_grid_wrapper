import sys
import os
from init_run import Setup
from parallel_worker import run_serial_job, collect_output
import shutil
from itertools import islice
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import numpy as np
import socket
from solar_abundances import periodic_table, solar_abundances
from solar_isotopes import solar_isotopes


def write_abund_file(tmp_dir):
    # file path is temp_dir + abund
    file_path = os.path.join(tmp_dir, "abund_to_use")
    # open file
    with open(file_path, "w") as file:
        # write the number of elements
        # write the elements and their abundances
        for element in periodic_table:
            if element != "":
                if element == "H" or element == "He":
                    abundance_to_write = solar_abundances[element]
                else:
                    abundance_to_write = solar_abundances[element]
                file.write(f"{element:<4} {abundance_to_write:>6.3f}\n")
    return file_path

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


def get_dask_client(client_type: str, cluster_name: str, workers_amount_cpus: int, nodes=1, slurm_script_commands=None,
                    slurm_memory_per_core=3.6, time_limit_hours=72, slurm_partition="debug", **kwargs):
    if cluster_name is None:
        cluster_name = "unknown"
    print("Preparing workers")
    if client_type == "local":
        client = get_local_client(workers_amount_cpus)
    elif client_type == "slurm":
        client = get_slurm_cluster(workers_amount_cpus, nodes, slurm_memory_per_core,
                                   script_commands=slurm_script_commands, time_limit_hours=time_limit_hours,
                                   slurm_partition=slurm_partition, **kwargs)
    else:
        raise ValueError("client_type must be either local or slurm")

    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {cluster_name} (change in config if not the case)")

    print(f"ssh -N -L {port}:{host}:{port} {cluster_name}")
    print(f"Then go to http://localhost:{port}/status to check the status of the workers")

    print("Worker preparation complete")

    return client


def get_local_client(workers_amount, **kwargs):
    if workers_amount >= 1:
        client = Client(threads_per_worker=1, n_workers=workers_amount, **kwargs)
    else:
        client = Client(threads_per_worker=1, **kwargs)
    return client


def get_slurm_cluster(cores_per_job: int, jobs_nodes: int, memory_per_core_gb: int, script_commands=None,
                      time_limit_hours=72, slurm_partition='debug', **kwargs):
    if script_commands is None:
        script_commands = [            # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
    # Create a SLURM cluster object
    # split into days, hours in format: days-hh:mm:ss
    days = time_limit_hours // 24
    hours = time_limit_hours % 24
    if days == 0:
        time_limit_string = f"{int(hours):02d}:00:00"
    else:
        time_limit_string = f"{int(days)}-{int(hours):02d}:00:00"
    print(time_limit_string)
    cluster = SLURMCluster(
        queue=slurm_partition,                      # Which queue/partition to submit jobs to
        cores=cores_per_job,                     # Number of cores per job (so like cores/workers per node)
        memory=f"{memory_per_core_gb * cores_per_job}GB",         # Amount of memory per job (also per node)
        job_script_prologue=script_commands,     # Additional commands to run before starting dask worker
        walltime=time_limit_string                      # Time limit for each job
    )
    cluster.scale(jobs=jobs_nodes)      # How many nodes
    client = Client(cluster)

    return client


def chunks(data, SIZE=1000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def mkdir(directory: str):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

def load_aux_data(file):
    atmos, feh, abunds = np.loadtxt(file, comments="#", usecols=(0, 3, 7), unpack=True, dtype=str)
    abunds = abunds.astype(float)
    feh = feh.astype(float)
    return atmos, feh, abunds

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def check_same_element_loc_in_two_arrays(array1, array2_float, elem1: str, elem2_float, str_to_add_array1):
    """
    Checks whether elem1 array1 is located in the same location as elem2 in array2. If not or if not located there at
    all, returns False.
    """
    tolerance_closest_abund = 0.001  # tolerance to consider abundances the same

    array1 = np.asarray(array1)
    array2 = np.asarray(array2_float)
    loc1 = np.where(array1 == f"'{elem1.replace(str_to_add_array1, '').replace('.mod', '').replace('/', '')}'")[0]
    #loc2_closest_index = find_nearest_index(array2, elem2_float)

    #print(array1, array2, elem1, elem2_float, f"'{elem1.replace(str_to_add_array1, '').replace('.mod', '')}'")

    if np.size(loc1) == 0:
        return False

    for index_to_check in loc1:
        #np.abs(array2[loc1[0]] - elem2_float) < tolerance_closest_abund
        if np.abs(array2[index_to_check] - elem2_float) < tolerance_closest_abund:
            return True
    return False



if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if len(sys.argv) > 2:
            aux_done_file = sys.argv[2]
            check_done_aux_files = True
        else:
            check_done_aux_files = False
            skip_fit = False
    else:
        config_file = './config.cfg'
        check_done_aux_files = False
        skip_fit = False

    """ Read config. file and distribute individual jobs """
    setup = Setup(file=config_file)
    jobs = setup.distribute_jobs()
    setup.atmos = None
    """ Start individual (serial) jobs in parallel """

    # create setup.common_wd if it does not exist
    if not os.path.exists(setup.common_wd):
        os.makedirs(setup.common_wd)
    # also create directories inside if they dont exist: /temp_directory_for_m3d/ and /temporary_grid/
    if not os.path.exists(setup.common_wd + '/temp_directory_for_m3d/'):
        os.makedirs(setup.common_wd + '/temp_directory_for_m3d/')
    if not os.path.exists(setup.common_wd + '/temporary_grid/'):
        os.makedirs(setup.common_wd + '/temporary_grid/')

    # copy atom file to common_wd and rename to atom.my_atom
    shutil.copy(os.path.join(setup.atom_path, setup.atom_id), os.path.join(setup.common_wd, 'atom.my_atom'))

    write_abund_file(setup.common_wd)
    write_isotope_file(setup.common_wd)

    login_node_address = setup.login_node_address

    slurm = setup.slurm_cluster
    if slurm:
        client = get_dask_client(client_type='slurm', cluster_name=login_node_address,
                                 workers_amount_cpus=setup.ncpu,
                                 nodes=setup.slurm_nodes, slurm_memory_per_core=3.6, time_limit_hours=setup.slurm_time_limit_hours,
                                 slurm_partition=setup.slurm_partition)
    else:
        client = get_dask_client(client_type='local', cluster_name=login_node_address,
                                 workers_amount_cpus=setup.ncpu,
                                 nodes=1, slurm_memory_per_core=3.6)

    #print("Creating temporary directories")

    """all_temporary_directories = []
    for i in range(setup.ncpu):
        all_temporary_directories.append(setup.common_wd + '/job_%03d/' % i)

    futures_test = []
    for temp_dir in all_temporary_directories:
        big_future = client.scatter([setup, temp_dir])
        future_test = client.submit(assign_temporary_directory, big_future)
        futures_test.append(future_test)
    futures_test = client.gather(futures_test)"""

    #for temp_dir in all_temporary_directories:
    #    setup_temp_dirs(setup, temp_dir)

    if check_done_aux_files:
        done_atmos, feh, done_abunds = load_aux_data(aux_done_file)
        done_abunds = done_abunds - feh

    print("Starting jobs")

    jobs_amount: int = 0

    #jobs_split = np.split(jobs, math.ceil(len(jobs) / 1000))

    MAX_TASKS_PER_CPU_AT_A_TIME = 16000

    all_futures_combined = []

    for one_jobs_split in chunks(jobs, setup.ncpu * MAX_TASKS_PER_CPU_AT_A_TIME):
        futures = []
        for one_job in one_jobs_split:
            #big_future = client.scatter(args[i])  # good
            if check_done_aux_files:
                abund, atmo = jobs[one_job].abund, jobs[one_job].atmo
                skip_fit = check_same_element_loc_in_two_arrays(done_atmos, done_abunds, atmo, abund, setup.atmos_path)

            if not skip_fit:
                jobs_amount += 1
                abund, atmo = jobs[one_job].abund, jobs[one_job].atmo
                #big_future_setup = client.scatter(setup, broadcast=True)
                #[big_future_setup] = client.scatter([setup], broadcast=True)

                #[fut_dict] = client.scatter([setup], broadcast=True)
                #score_guide = lambda row: expensive_computation(fut_dict, row)
                # job, temporary_directory, m3dis_path, convlim, iterations_max, use_absmet, absmet_global_path, hash_table_size
                future = client.submit(run_serial_job, abund, atmo, setup.atmos_path, setup.common_wd, setup.m3d_path, setup.conv_lim,
                                       setup.max_iterations, setup.use_absmet, setup.absmet_global_path, setup.hash_table_size)
                futures.append(future)  # prepares to get values

        print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = client.gather(futures)  # starts the calculations (takes a long time here)
        print("Worker calculation done")  # when done, save values
        all_futures_combined += futures

    client.close()

    #setup.njobs = jobs_amount
    collect_output(setup.common_wd + '/temporary_grid/',  setup.atom_id, setup.atom_comment, jobs_amount)
