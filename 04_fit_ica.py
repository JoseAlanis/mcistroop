"""
==============================
Independent Component Analysis
==============================

Decompose EEG signal into independent components

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

import matplotlib.pyplot as plt

from mne import open_report
from mne.io import read_raw_fif
from mne.preprocessing import ICA

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat, n_jobs

# Handle command line arguments
args = parser.parse_args()
subject = args.subject
session = args.session
task = args.task

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Fit ICA for subject %s (%s)' % (subject, task) +
      LoggingFormat.END)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          task=task,
                          processing_step='repairbads',
                          file_type='raw.fif')

# check if file exists, otherwise terminate the script
if not os.path.isfile(input_file) or subject == 60:
    exit()

# import data
raw = read_raw_fif(input_file, preload=True)

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
ica_fig = ica.plot_components(picks=range(0, 5), show=False)
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
with open_report(fname.report(subject=subject, task=task)[0]) as report:
    report.add_figs_to_section(ica_fig, 'ICA solution',
                               section='ICA',
                               replace=True)
    report.save(fname.report(subject=subject, task=task)[1], overwrite=True,
                open_browser=False)
