import sys
import os
import numpy as np
import shutil
# local
import pickle
from configparser import ConfigParser
import shutil


def mkdir(directory: str):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


class SerialJob:
    def __init__(self, i: int):
        self.id = i
        #self.tmp_wd = parent.common_wd + '/job_%03d/' % self.id
        self.atmo: str = None
        self.abund: float = None
        self.output = {}



# a setup of the run to compute NLTE grid, e.g. Mg over all MARCS grid
class Setup:
    def __init__(self, file='config.cfg'):
        config = ConfigParser()

        config.read(file)

        self.common_wd = config.get('Parameters', 'common_wd')
        self.atmos_list = config.get('Parameters', 'atmos_list')
        self.atmos_format = config.get('Parameters', 'atmos_format')
        self.atmos_path = config.get('Parameters', 'atmos_path')
        self.atom_path = config.get('Parameters', 'atom_path')
        self.atom_id = config.get('Parameters', 'atom_id')
        self.atom_comment = config.get('Parameters', 'atom_comment')
        self.m3d_path = config.get('Parameters', 'm3d_path')
        self.ncpu = config.getint('Parameters', 'ncpu')
        self.start_abund = config.getfloat('Parameters', 'start_abund')
        self.end_abund = config.getfloat('Parameters', 'end_abund')
        self.step_abund = config.getfloat('Parameters', 'step_abund')
        self.slurm_cluster = config.get('Parameters', 'slurm_cluster')
        self.slurm_partition = config.get('Parameters', 'slurm_partition')
        self.slurm_time_limit_hours = config.getint('Parameters', 'slurm_time_limit_hours')
        self.slurm_nodes = config.getint('Parameters', 'slurm_nodes')
        self.login_node_address = config.get('Parameters', 'login_node_address')
        self.use_absmet = self._convert_string_to_bool(config.get('Parameters', 'use_absmet'))
        self.absmet_global_path = config.get('Parameters', 'absmet_global_path')
        self.hash_table_size = config.getint('Parameters', 'hash_table_size')
        self.max_iterations = config.getint('Parameters', 'max_iterations')
        self.conv_lim = config.getfloat('Parameters', 'conv_lim')


        self.slurm_cluster = self._convert_string_to_bool(self.slurm_cluster)

        # convert common_wd to absolute path
        if self.common_wd.startswith('./'):
            self.common_wd = os.path.join(os.getcwd(), self.common_wd[2:])
        if self.m3d_path.startswith('./'):
            self.m3d_path = os.path.join(os.getcwd(), self.m3d_path[2:])



        """
        Read *a list* of all requested model atmospheres
        Add a path to the filenames
        Model atmospheres themselves are not read here,
        as the parallel worker will iterate over them
        """
        print(f"Reading a list of model atmospheres from {self.atmos_list}")
        atmos_list = np.loadtxt(self.atmos_list, dtype=str, ndmin=1)
        self.atmos = []
        for atm in atmos_list:
                self.atmos.append(os.path.join(self.atmos_path, atm))

    @staticmethod
    def _convert_string_to_bool(string_to_convert: str) -> bool:
        if string_to_convert.lower() in ["true", "yes", "y", "1"]:
            return True
        elif string_to_convert.lower() in ["false", "no", "n", "0"]:
            return False
        else:
            raise ValueError(f"Configuration: could not convert {string_to_convert} to a boolean")

    def distribute_jobs(self):
        """
        Distributing model atmospheres over a number of processes
        input:
        (array) atmos_list: contains all model atmospheres requested for the run
        (integer) ncpu: number of CPUs to use
        """
        print(50 * "-")
        print(f"Distributing model atmospheres over {self.ncpu} CPUs")

        atmos_list = self.atmos

        """
        abundance dimension:
        every NLTE run with M1D has unique model atmospehere,
                model atom and abundance of the NLTE element
        assuming one run is set up for one NLTE element,
        one needs to iterate over model atmospheres and abundances
        """
        abund_list = np.arange(self.start_abund, self.end_abund, self.step_abund)
        # [start, end) --> [start, end]
        abund_list = np.hstack((abund_list, self.end_abund))

        totn_jobs = len(atmos_list) * len(abund_list)
        #self.njobs = totn_jobs
        print('total # jobs', totn_jobs)

        atmos_list, abund_list = np.meshgrid(atmos_list, abund_list)
        atmos_list = atmos_list.flatten()
        abund_list = abund_list.flatten()

        jobs = {}

        #job.atmos = atmos_list
        #job.abund = abund_list

        for i, (one_atmo, one_abund) in enumerate(zip(atmos_list, abund_list)):
            jobs[i] = SerialJob(i)
            #self.jobs[i].id = i
            jobs[i].atmo = one_atmo
            jobs[i].abund = one_abund

        return jobs
