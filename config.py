"""
========================
Study configuration file
========================

Configuration parameters and global variable values for the study.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os
from os import path as op
import platform
import multiprocessing

from argparse import ArgumentParser

from mne.channels import make_standard_montage

from utils import FileNames

# get current working directory and its parent directory
wd = os.path.abspath(os.getcwd())
root_path = os.path.dirname(wd)

# set relative input and output paths for generating the sourcedata directory
input_path = os.path.join(root_path, 'orig_eeg')
output_path = os.path.join(root_path, 'data_bids')

# tasks
tasks = ['congruentstroop', 'incongruentstroop', 'mixedstroop']

event_ids = {
    # fix cross
    'Bedingung/B100': 100,

    # start of tasks
    'Bedingung/B 21': 501,
    'Bedingung/B 23': 502,
    'Bedingung/B 35': 503,

    # word stimuli
    'Bedingung/B 61': 101,
    'Bedingung/B 62': 102,
    'Bedingung/B 63': 103,
    'Bedingung/B 64': 104,

    'Bedingung/B 71': 201,
    'Bedingung/B 72': 202,
    'Bedingung/B 73': 203,
    'Bedingung/B 74': 204,

    'Bedingung/B 81': 301,
    'Bedingung/B 82': 302,
    'Bedingung/B 83': 303,
    'Bedingung/B 84': 304,

    'Bedingung/B 91': 401,
    'Bedingung/B 92': 402,
    'Bedingung/B 93': 403,
    'Bedingung/B 94': 404,

    # responses
    'Bedingung/B  5': 1,
    'Bedingung/B  6': 2,
    'Bedingung/B  7': 3,
    'Bedingung/B  8': 4}

# get sensors arrangement
montage = make_standard_montage(kind='standard_1005')

# threshold for rejection (peak to peak amplitude)
max_peak = 200e-6

class LoggingFormat:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


###############################################################################
# User parser to handle command line arguments
parser = ArgumentParser(description='Parse command line argument for '
                                    'pre-processing of EEG data.')
parser.add_argument('-s', '--subject',
                    # metavar='sub-\\d+',
                    help='The subject to process (int)',
                    type=int)
parser.add_argument('-ses', '--session',
                    # metavar='E|M',
                    help='The session to process (str)',
                    type=str)
parser.add_argument('-t', '--task',
                    # metavar='E|M',
                    help='The task to process (str)',
                    type=str)

# Determine which user is running the scripts on which machine. Set the path to
# where the data is stored and determine how many CPUs to use for analysis.
node = platform.node()  # Maschine
system = platform.system()  # Os

# You want to add your machine to this list
if 'Jose' in node and 'n' in system:
    # iMac at work
    data_dir = '../data_bids'
    n_jobs = 2  # This station has 4 cores (we'll use 2).
elif 'jose' in node and 'x' in system:
    # pc at home
    data_dir = '../data_bids'
    n_jobs = 'cuda'  # Use NVIDIA CUDA GPU processing
else:
    # Defaults
    data_dir = '../data'
    n_jobs = 1

# For BLAS to use the right amount of cores
use_cores = multiprocessing.cpu_count()//2
if use_cores < 2:
    use_cores = 1
os.environ['OMP_NUM_THREADS'] = str(use_cores)


###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# directories to use for input and output
fname.add('data_dir', data_dir)

fname.add('derivatives_dir', '{data_dir}/derivatives')
fname.add('reports_dir', '{derivatives_dir}/reports')
fname.add('results', '{derivatives_dir}/results')
fname.add('rt', '{results}/rt')
fname.add('figures', '{results}/figures')
fname.add('tables', '{results}/tables')


# The paths for data file input
# alternative:
def source_file(files, subject, task, data_type):
    return \
        op.join(files.data_dir,
                'sub-%03d/%s/'
                'sub-%03d_task-%s_%s.vhdr' % (subject,
                                              data_type,
                                              subject,
                                              task,
                                              data_type))


# create full path for data file input
fname.add('source', source_file)


# The paths that are produced by the analysis steps
def output_file(path, processing_step, subject, task, file_type):
    path = op.join(path.derivatives_dir, processing_step, 'sub-%03d' % subject)
    os.makedirs(path, exist_ok=True)
    return op.join(path,
                   'sub-%03d_task-%s_%s-%s' % (subject,
                                               task,
                                               processing_step,
                                               file_type))


# The full path for data file output
fname.add('output', output_file)


# The paths that are produced by the report step
def report_path(path, subject, task):
    h5_path = op.join(path.reports_dir, 'sub-%03d-%s.h5' % (subject, task))
    html_path = op.join(path.reports_dir, 'sub-%03d-%s-report.html' % (subject, task))  # noqa
    return h5_path, html_path


# The full path for report file output
fname.add('report', report_path)

# Files produced by system check and validator
fname.add('system_check', './system_check.txt')
fname.add('validator', './validator.txt')
