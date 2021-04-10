"""
==============================
Independent Component Analysis
==============================

Decompose EEG signal into independent components

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

import numpy as np

import matplotlib.pyplot as plt

from mne import open_report, create_info
from mne.io import read_raw_fif, RawArray
from mne.preprocessing import ICA

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat, n_jobs, montage

# Handle command line arguments
args = parser.parse_args()
subject = args.subject
session = args.session
task = args.task

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Fit ICA for subject %s' % subject +
      LoggingFormat.END)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          task=task,
                          processing_step='repairbads',
                          file_type='raw.fif')

# check if file exists, otherwise terminate the script
if not os.path.isfile(input_file):
    exit()

# import data
raw = read_raw_fif(input_file, preload=True)
raw.apply_proj()

###############################################################################
# 2) Create placeholders for interpolation of missing channels
custom_info = create_info(ch_names=['FCz', 'Cz'],
                          ch_types=['eeg', 'eeg'],
                          sfreq=raw.info['sfreq'])

custom_data = np.zeros((len(custom_info['ch_names']),
                        raw.get_data().shape[1]))

custom_raw = RawArray(custom_data, custom_info, raw.first_samp)
custom_raw.info['highpass'] = raw.info['highpass']
custom_raw.info['lowpass'] = raw.info['lowpass']

###############################################################################
# 3) Add newly created channels to original raw
raw.add_channels([custom_raw])
raw.set_montage(montage=montage)

# interpolate the added channels
raw.info['bads'] = ['FCz', 'Cz']
raw.interpolate_bads(mode='accurate')

# filter data to remove drifts
raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=n_jobs)

###############################################################################
#  2) Set ICA parameters
n_components = 5
method = 'infomax'
reject = dict(eeg=250e-6)

###############################################################################
#  2) Fit ICA
ica = ICA(n_components=n_components,
          method=method,
          fit_params=dict(extended=True))

ica.fit(raw_filt,
        reject=reject,
        reject_by_annotation=True)

###############################################################################
# 3) Plot ICA components
ica_fig = ica.plot_components(picks=range(0, 15), show=False)
plt.close('all')

###############################################################################
# 4) Save ICA solution
# output path
output_path = fname.output(subject=subject,
                           task=task,
                           processing_step='fitica',
                           file_type='ica.fif')
# save file
ica.save(output_path)

###############################################################################
# 5) Create HTML report
with open_report(fname.report(subject=subject)[0]) as report:
    report.add_figs_to_section(ica_fig, 'ICA solution',
                               section='ICA',
                               replace=True)
    report.save(fname.report(subject=subject)[1], overwrite=True,
                open_browser=False)
