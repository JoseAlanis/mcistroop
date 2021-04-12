"""
====================
EEG data set to BIDS
====================

Put EEG data files into a BIDS-compliant directory structure.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

# import matplotlib.pyplot as plt

import pandas as pd

from mne import events_from_annotations, concatenate_raws, Annotations
from mne.io import read_raw_brainvision, read_raw_fif

from mne_bids import write_raw_bids, BIDSPath

from config import input_path, output_path, montage, tasks, event_ids

###############################################################################
# 1) check if target directory exists
sourcedata_path = os.path.join(output_path, 'sourcedata')
os.makedirs(sourcedata_path, exist_ok=True)

# get subject id and file names
ids = pd.read_csv('../data_bids/subject_data/subject_ids.tsv',
                  sep='\t',
                  header=0)

###############################################################################
# 2) loop through files and crate copy of the file in question, extract the
# necessary information and export the copy in a way BIDS-compliant way
for index, row in ids.iterrows():

    # bad files
    if row['file_name'] in {'03-08-E.vhdr', '14_08_M.vhdr'} :
        continue

    print(index, row)

    # path and filename
    fname = os.path.join(input_path, row['file_name'])

    # *** 1) import file and extract file info ***
    raw = read_raw_brainvision(fname, preload=False)
    # sampling rate
    sfreq = raw.info['sfreq']
    # event onsets and events ids (annotations entailed in the EEG-file)
    events, event_ids = events_from_annotations(raw, event_id=event_ids)

    # *** 2) get start and end event markers for the desired tasks ***
    # name of fix cross end marker
    start_marker = 0
    fix_marker = event_ids['Bedingung/B100']
    for task in tasks:

        if task == 'congruentstroop':
            # name of start marker
            start_marker = event_ids['Bedingung/B 21']
        elif task == 'incongruentstroop':
            # name of start marker
            start_marker = event_ids['Bedingung/B 23']
        elif task == 'mixedstroop':
            # name of start marker
            start_marker = event_ids['Bedingung/B 35']

        # latencies for start and end of task
        starts = events[events[:, 2] == start_marker, 0]

        # check if start markers were send twice (i.e., one shortly after
        # the other)
        correct_starts = [s0 for s0, s1 in zip(starts, starts[1:])
                          if s1-s0 > 10*sfreq]
        correct_starts.append(starts[-1])

        # *** 3) extract segments of data corresponding to task in question ***
        cs_raws = []
        # look for consecutive stimuli in the recording
        for start in correct_starts:
            # first event in run
            tmin = (start / sfreq) - 5.0

            # look for last event in run by indentifiying long breaks during
            # the recording
            fix_cross_evs = \
                events[(events[:, 0] > start) & (events[:, 2] == fix_marker)]

            i = 0
            while (fix_cross_evs[i+1, 0] - fix_cross_evs[i, 0]) < 15*sfreq:
                i += 1
                if fix_cross_evs[i, 0] == fix_cross_evs[-1, 0]:
                    break

            # add some seconds to avoid excluding late responses
            tmax = (fix_cross_evs[i, 0] / sfreq) + 8.0

            # put segments of data in list
            cs_raws.append(raw.copy().crop(tmin=tmin, tmax=tmax))

        # concatenate segments of data
        cs_raws = concatenate_raws(cs_raws)

        # *** 4) modify dataset info  ***
        # identify channel types based on matching names in montage
        types = []
        channels = cs_raws.info['ch_names']
        for chan in channels:
            if chan in montage.ch_names:
                types.append('eeg')
            elif chan.startswith('EOG') | chan.startswith('EXG'):
                types.append('eog')
            else:
                types.append('stim')

        # add channel types and eeg-montage
        cs_raws.set_channel_types(
            {chan: typ for chan, typ in zip(channels, types)})
        cs_raws.set_montage(montage=montage)

        # frequency of power line
        cs_raws.info['line_freq'] = 50.0
        cs_raws.info['lowpass'] = sfreq / 2

        # *** 5) export task data ***
        # get subject and session ids for file in question
        subject = row['subject_id']

        # 5) check if target directory exists
        subj = 'sub-%s' % str(subject).rjust(3, '0')
        subject_path = os.path.join(output_path, 'sourcedata', subj, 'eeg')
        os.makedirs(subject_path, exist_ok=True)

        # *** 6) create exploratory plots  ***
        # plot data
        # raw_plot = cs_raws.plot(scalings=dict(eeg=50e-6, eog=50e-6),
        #                         n_channels=len(channels),
        #                         show=False)
        # raw_plot.set_size_inches(8.0, 6.0)
        # raw_plot.savefig(subject_path + '/%s_%s_data.png' % (subj, task),
        #                  dpi=300)

        # plot power spectral density
        # fig, ax = plt.subplots(figsize=(8, 4))
        # cs_raws.plot_psd(show=False, ax=ax)
        # fig.savefig(subject_path + '/%s_%s_psd.png' % (subj, task), dpi=300)
        # plt.close('all')

        # *** 7) save file ***
        output_fname = os.path.join(subject_path,
                                    '%s_task-%s-raw.fif' % (subj, task))
        cs_raws.save(output_fname, overwrite=True)

        # reload raw and export to bids
        raw_short = read_raw_fif(output_fname, preload=False)
        evs_short, ev_ids_short = events_from_annotations(raw_short,
                                                          event_id=event_ids)
        annotations = Annotations([], [], [])
        raw_short.set_annotations(annotations)

        # *** 8) save file in BIDS compliant format ***
        bids_path = BIDSPath(
            subject=str(subject).rjust(3, '0'),
            task=task,
            root=output_path,
            extension='.vhdr')

        # fixed names for events
        new_ids = {
            # fix cross
            'fix cross': 100,

            # start of tasks
            'start S/C': 501,
            'start S/I': 502,
            'start S/M': 503,

            'C/C/G': 101,
            'I/C/G': 102,
            'M/C/G': 103,
            'M/I/G': 104,

            'C/C/R': 201,
            'I/C/R': 202,
            'M/C/R': 203,
            'M/I/R': 204,

            'C/C/Y': 301,
            'I/C/Y': 302,
            'M/C/Y': 303,
            'M/I/Y': 304,

            'C/C/B': 401,
            'I/C/B': 402,
            'M/C/B': 403,
            'M/I/B': 404,

            # responses
            'G': 1,
            'R': 2,
            'Y': 3,
            'B': 4}

        # save file
        write_raw_bids(raw_short,
                       bids_path,
                       events_data=evs_short,
                       event_id=new_ids,
                       overwrite=True)
