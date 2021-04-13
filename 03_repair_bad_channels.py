"""
===================
Repair bad channels
===================

Identify and interpolate bad (i.e., noisy) EEG channels.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mne import Annotations, open_report, create_info
from mne.io import RawArray

from mne_bids import BIDSPath, read_raw_bids

# All parameters are defined in config.py
from config import parser, LoggingFormat, fname, output_bids, \
    n_jobs, montage, max_peak
from bads import find_bad_channels
from viz import plot_z_scores

# Handle command line arguments
args = parser.parse_args()
subject = args.subject
session = args.session
task = args.task

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Bad channel detection for subject %s (%s)' % (subject, task) +
      LoggingFormat.END)

##############################################################################
# 1) import the data

# name of the file
input_file = fname.source(subject=subject,
                          task=task,
                          data_type='eeg')
# bids-formatted path to data
bids_path = BIDSPath(
    subject=str(subject).rjust(3, '0'),
    task=task,
    root=output_bids,
    extension='.vhdr')

# check if file exists, otherwise terminate the script
if not os.path.isfile(input_file):
    exit()

# import existing file
# raw = read_raw_brainvision(input_file, preload=True)
raw = read_raw_bids(bids_path, extra_params=dict(preload=True))
raw.set_montage(montage)

##############################################################################
# 2) Remove slow drifts and line noise

# Setting up band-pass filter from 0.1 - 40 Hz
#
# FIR filter parameters
# ---------------------
# Designing a one-pass, zero-phase, non-causal bandpass filter:
# - Windowed time-domain design (firwin) method
# - Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
# - Lower passband edge: 0.10
# - Lower transition bandwidth: 0.10 Hz (-6 dB cutoff frequency: 0.05 Hz)
# - Upper passband edge: 40.00 Hz
# - Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)
# - Filter length: 33001 samples (33.001 sec)
raw = raw.copy().filter(l_freq=0.1, h_freq=40.,
                        picks=['eeg', 'eog'],
                        filter_length='auto',
                        l_trans_bandwidth='auto',
                        h_trans_bandwidth='auto',
                        method='fir',
                        phase='zero',
                        fir_window='hamming',
                        fir_design='firwin',
                        n_jobs=n_jobs)

##############################################################################
# 3) Find noisy channels and compute robust average reference
sfreq = raw.info['sfreq']
channels = raw.copy().pick_types(eeg=True).ch_names

# extract eeg signal
eeg_signal = raw.get_data(picks='eeg')

# reference signal to robust estimate of central tendency
ref_signal = np.nanmedian(eeg_signal, axis=0)

i = 0
noisy = []
while True:
    # remove reference
    eeg_temp = eeg_signal - ref_signal

    # find bad channels by deviation (high variability in amplitude)
    bad_dev = find_bad_channels(eeg_temp,
                                channels=channels,
                                method='deviation')['deviation']

    # find channels that don't well with other channels
    bad_corr = find_bad_channels(eeg_temp,
                                 channels=channels,
                                 sfreq=sfreq,
                                 r_threshold=0.4,
                                 percent_threshold=0.05,
                                 time_step=1.0,
                                 method='correlation')['correlation']

    # only keep unique values
    bads = set(bad_dev) | set(bad_corr)

    # save identified noisy channels
    if bads:
        noisy.extend(bads)
        print('Found bad channels %s'
              % (', '.join([str(chan) for chan in bads])))

        # interpolate noisy channels
        raw_copy = raw.copy()
        raw_copy.info['bads'] = noisy
        raw_copy.interpolate_bads(mode='accurate')
        eeg_signal = raw_copy.get_data(picks='eeg')

    # compute new reference (mean of signal with interpolated channels)
    ref_signal = np.nanmean(eeg_signal, axis=0)

    # break if no (more) bad channels found
    if (i > 0 and len(bads) == 0) or i > 4:
        print('Finishing after i == %s' % i)
        break

    i = i + 1

##############################################################################
# 4) Compute robust average reference for EEG data
# remove robust reference
eeg_signal = raw.get_data(picks='eeg')
eeg_temp = eeg_signal - ref_signal

# bad by (un)correlation
bad_corr = find_bad_channels(eeg_temp,
                             channels=channels,
                             sfreq=sfreq,
                             r_threshold=0.4,
                             percent_threshold=0.05,
                             time_step=1.0,
                             method='correlation')['correlation']

# bad by deviation
bad_dev = find_bad_channels(eeg_temp,
                            channels=channels,
                            method='deviation',
                            return_z_scores=True)

z_scores = bad_dev['deviation_z_scores']
bad_dev = bad_dev['deviation']

# only keep unique values
bad_channels = set(bad_dev) | set(bad_corr)

# create plot showing channels z-scores
fig = plot_z_scores(z_scores, channels=channels, bads=bad_channels, show=False)

# interpolate channels identified by deviation criterion
raw.info['bads'] = list(bad_channels)
raw.interpolate_bads(mode='accurate')

##############################################################################
# 5) Reference eeg data to average of all eeg channels
raw.set_eeg_reference(ref_channels='average', projection=True)

##############################################################################
# 6) Find distorted segments in data
# channels to use in artefact detection procedure
eeg_channels = raw.copy().pick_types(eeg=True).ch_names

# ignore fronto-polar channels
picks = [raw.ch_names.index(channel)
         for channel in eeg_channels if channel not in
         {'AFp1', 'AFp2', 'AFF5h', 'AFF1h', 'AFF2h', 'AFF6h'}]

# use a copy of eeg data
raw_copy = raw.copy()
raw_copy.apply_proj()
data = raw_copy.get_data(eeg_channels)

# detect artifacts (i.e., absolute amplitude > 500 microV)
times = []
annotations_df = pd.DataFrame(times)
onsets = []
duration = []
annotated_channels = []
bad_chans = []

# loop through samples
for sample in range(0, data.shape[1]):
    if len(times) > 0:
        if sample <= (times[-1] + int(1 * sfreq)):
            continue
    peak = []
    for channel in picks:
        peak.append(abs(data[channel][sample]))
    if max(peak) >= max_peak:
        times.append(float(sample))
        annotated_channels.append(raw_copy.ch_names[picks[int(np.argmax(
            peak))]])
# if artifact found create annotations for raw data
if len(times) > 0:
    # get first time
    first_time = raw_copy.first_time
    # column names
    annot_infos = ['onset', 'duration', 'description']

    # save onsets
    onsets = np.asarray(times)
    # include one second before artifact onset
    onsets = ((onsets / sfreq) + first_time) - 1
    # durations and labels
    duration = np.repeat(2, len(onsets))
    description = np.repeat('Bad', len(onsets))

    # get annotations in data
    artifacts = np.array((onsets, duration, description)).T
    # to pandas data frame
    artifacts = pd.DataFrame(artifacts,
                             columns=annot_infos)
    # annotations from data
    annotations = pd.DataFrame(raw_copy.annotations)
    annotations = annotations[annot_infos]

    # merge artifacts and previous annotations
    artifacts = artifacts.append(annotations, ignore_index=True)

    # create new annotation info
    annotations = Annotations(artifacts['onset'],
                              artifacts['duration'],
                              artifacts['description'])
    # apply to raw data
    raw.set_annotations(annotations)

# save total annotated time
total_time = sum(duration)
# save frequency of annotation per channel
frequency_of_annotation = {x: annotated_channels.count(x) * 2
                           for x in annotated_channels}

# create plot with clean data
plot_artefacts = raw.plot(scalings=dict(eeg=50e-6, eog=50e-6),
                          n_channels=len(raw.info['ch_names']),
                          title='Robust reference applied to Sub-%s' % subject,
                          show=False)
plt.close('all')

##############################################################################
# 7) Add the computed robust average reference to the data
raw.apply_proj()

###############################################################################
# 8) Create and interpolation missing channels

# place holder structure
custom_info = create_info(ch_names=['FCz', 'Cz'],
                          ch_types=['eeg', 'eeg'],
                          sfreq=raw.info['sfreq'])

custom_data = np.zeros((len(custom_info['ch_names']),
                        raw.get_data().shape[1]))

custom_raw = RawArray(custom_data, custom_info, raw.first_samp)
custom_raw.info['highpass'] = raw.info['highpass']
custom_raw.info['lowpass'] = raw.info['lowpass']
custom_raw.info['line_freq'] = raw.info['line_freq']

# add newly created channels to original raw
raw.add_channels([custom_raw])
raw.set_montage(montage=montage)

# interpolate the added channels using original data
raw.info['bads'] = ['FCz', 'Cz']
raw.interpolate_bads(mode='accurate')

##############################################################################
# 9) Export data to .fif for further processing
# output path
output_path = fname.output(subject=subject,
                           task=task,
                           processing_step='repairbads',
                           file_type='raw.fif')

# save file
raw.save(output_path, overwrite=True)

# cretae fisrt report
with open_report(fname.report(subject=subject, task=task)[0]) as report:
    report.parse_folder(os.path.dirname(output_path),
                        pattern=os.path.basename(output_path),
                        render_bem=False)
    report.add_figs_to_section(fig, 'Robust Z-Scores',
                               section='Bad channel detection',
                               replace=True)
    report.add_figs_to_section(plot_artefacts,
                               'Artefacts',
                               section='Raw data',
                               replace=True)
    report.save(fname.report(subject=subject, task=task)[1], overwrite=True,
                open_browser=False)
