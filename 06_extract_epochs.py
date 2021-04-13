"""
==================================
Extract epochs from continuous EEG
==================================

Extract epochs for each experimental condition

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

import pandas as pd
import numpy as np

from mne import events_from_annotations, Epochs
from mne.io import read_raw_fif

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat

# Handle command line arguments
args = parser.parse_args()
subject = args.subject
session = args.session
task = args.task

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Extracting epochs for subject %s (%s)' % (subject, task) +
      LoggingFormat.END)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          task=task,
                          processing_step='cleaned',
                          file_type='raw.fif')

# check if file exists, otherwise terminate the script
if not os.path.isfile(input_file):
    exit()

# import data
raw = read_raw_fif(input_file, preload=True)

# only keep EEG channels
raw.pick_types(eeg=True)

###############################################################################
# 2) Get events from continuous EEG data
ev_id = None
# if task in {'congruentstroop', 'incogruentstroop'}:
#     # create a dictionary with event IDs for standardised handling
#     ev_id = {'Stimulus/S 12': 100,
#
#              'Stimulus/S  8': 1,
#              'Stimulus/S  9': 2,
#              'Stimulus/S 10': 3,
#              'Stimulus/S 11': 4,
#
#              'Stimulus/S  2': 5,
#              'Stimulus/S  3': 6,
#              'Stimulus/S  4': 7,
#              'Stimulus/S  5': 8,
#              }
# elif task == {'mixedstroop'}:
#     # create a dictionary with event IDs for standardised handling
#     ev_id = {'Stimulus/S 16': 100,
#
#              'Stimulus/S  8': 1,
#              'Stimulus/S  9': 101,
#
#              'Stimulus/S 10': 2,
#              'Stimulus/S 11': 102,
#
#              'Stimulus/S 12': 3,
#              'Stimulus/S 13': 103,
#
#              'Stimulus/S 14': 4,
#              'Stimulus/S 15': 104,
#
#              'Stimulus/S  2': 5,
#              'Stimulus/S  3': 6,
#              'Stimulus/S  4': 7,
#              'Stimulus/S  5': 8,
#              }
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


# extract events
events = events_from_annotations(raw, event_id=new_ids, regexp=None)

# events[0][events[0][:, 2] == 100].shape

###############################################################################
# 3) Recode events into respective conditions and add information about valid
# and invalid responses

# copy of events
new_evs = events[0].copy()

# global variables
sfreq = raw.info['sfreq']
block_end = events[0][events[0][:, 2] == 17, 0] / sfreq

# trial counter
fix = 0
trial = 0
color = []
condition = []
valid_trial = True
invalid_trials = []
good_trials = []
miss = []


# place holders for results
block = []
reaction = []
rt = []

# loop trough events and recode them
for event in range(len(new_evs[:, 2])):

    # # save block based on onset (before or after break)
    # if (new_evs[event, 0] / sfreq) < block_end:
    #     block.append(0)
    # else:
    #     block.append(1)

    # - if event is a fix cross -
    if new_evs[event, 2] == 100:
        # -- if fix cross was followed by word stimulus --
        if new_evs[event + 1, 2] in {1, 2, 3, 4, 101, 102, 103, 104}:
            # change stimulus id to mark the beginning of a valid trial,
            # store stimulus id
            new_evs[event, 2] = 200
            valid_trial = True
        else:
            # change stimulus id to mark the beginning of an invalid trial,
            # (e.g., subject pressed the button too soon), store stimulus id
            new_evs[event, 2] = 300
            valid_trial = False
            invalid_trials.append(fix)

        # proceed to next stimulus
        fix += 1
        continue

    # - if event is a word stimulus -
    elif new_evs[event, 2] in {1, 2, 3, 4, 101, 102, 103, 104}:

        # -- if event is the last one --
        if new_evs[event, 0] == new_evs[-1, 0]:
            # skip as no reaction was provided
            new_evs[event, 2] = 99
            # add event as missed reaction
            miss.append(trial)
            reaction.append(np.nan)
            rt.append(np.nan)
            trial += 1

            if task == 'congruentstroop':
                condition.append('congruent')
            elif task == 'incongruentstroop':
                condition.append('incongruent')
            elif task == 'mixedstroop':
                if new_evs[event, 2] in {1, 2, 3, 4}:
                    condition.append('congruent')
                elif new_evs[event, 2] in {101, 102, 103, 104}:
                    condition.append('incongruent')

            # proceed to next stimulus
            continue

        # -- if event is followed by a reaction but trial is invalid --
        elif new_evs[event + 1, 2] in {5, 6, 7, 8} \
                and not valid_trial:
            miss.append(trial)
            reaction.append(np.nan)
            rt.append(np.nan)
            trial += 1

            if task == 'congruentstroop':
                condition.append('congruent')
            elif task == 'incongruentstroop':
                condition.append('incongruent')
            elif task == 'mixedstroop':
                if new_evs[event, 2] in {1, 2, 3, 4}:
                    condition.append('congruent')
                elif new_evs[event, 2] in {101, 102, 103, 104}:
                    condition.append('incongruent')

            # proceed to next stimulus
            continue

        # -- if event is followed by a reaction and trial is valid --
        elif new_evs[event + 1, 2] in {5, 6, 7, 8} \
                and valid_trial:

            # --- if color green and reaction green ---
            if new_evs[event, 2] in {1, 101}:
                # add color and condition
                color.append('green')
                if task == 'congruentstroop':
                    condition.append('congruent')
                elif task == 'incongruentstroop':
                    condition.append('incongruent')
                elif task == 'mixedstroop':
                    if new_evs[event, 2] in {1}:
                        condition.append('congruent')
                    elif new_evs[event, 2] in {101}:
                        condition.append('incongruent')

                if new_evs[event + 1, 2] == 5:
                    # change event id as correct
                    new_evs[event, 2] = 11
                    new_evs[event + 1, 2] = 21
                    reaction.append('correct')
                else:
                    new_evs[event, 2] = 31
                    reaction.append('incorrect')

            # --- if color red reaction red ---
            elif new_evs[event, 2] in {2, 102}:
                color.append('red')
                if task == 'congruentstroop':
                    condition.append('congruent')
                elif task == 'incongruentstroop':
                    condition.append('incongruent')
                elif task == 'mixedstroop':
                    if new_evs[event, 2] in {2}:
                        condition.append('congruent')
                    elif new_evs[event, 2] in {102}:
                        condition.append('incongruent')

                if new_evs[event + 1, 2] == 6:
                    # recode as correct
                    new_evs[event, 2] = 12
                    new_evs[event + 1, 2] = 22
                    reaction.append('correct')
                else:
                    new_evs[event, 2] = 32
                    reaction.append('incorrect')

            # --- if color yellow reaction yellow ---
            elif new_evs[event, 2] in {3, 103}:
                color.append('yellow')
                if task == 'congruentstroop':
                    condition.append('congruent')
                elif task == 'incongruentstroop':
                    condition.append('incongruent')
                elif task == 'mixedstroop':
                    if new_evs[event, 2] in {3}:
                        condition.append('congruent')
                    elif new_evs[event, 2] in {103}:
                        condition.append('incongruent')

                if new_evs[event + 1, 2] == 7:
                    # recode as correct
                    new_evs[event, 2] = 13
                    new_evs[event + 1, 2] = 23
                    reaction.append('correct')
                else:
                    new_evs[event, 2] = 33
                    reaction.append('incorrect')

            # --- if color blue reaction blue ---
            elif new_evs[event, 2] in {4, 104}:
                color.append('blue')
                if task == 'congruentstroop':
                    condition.append('congruent')
                elif task == 'incongruentstroop':
                    condition.append('incongruent')
                elif task == 'mixedstroop':
                    if new_evs[event, 2] in {4}:
                        condition.append('congruent')
                    elif new_evs[event, 2] in {104}:
                        condition.append('incongruent')

                if new_evs[event + 1, 2] == 8:
                    # recode as correct
                    new_evs[event, 2] = 14
                    new_evs[event + 1, 2] = 24
                    reaction.append('correct')
                else:
                    new_evs[event, 2] = 34
                    reaction.append('incorrect')

            # store trial id as good trial
            good_trials.append(trial)
            r_time = (new_evs[event + 1, 0] - new_evs[event, 0]) / sfreq
            rt.append(r_time)
            trial += 1
            continue

        # -- if event is a word but no reaction followed
        elif new_evs[event + 1, 2] not in {5, 6, 7, 8} \
                and valid_trial:

            # store trial as miss
            new_evs[event, 2] = 99
            miss.append(trial)
            trial += 1
            reaction.append(np.nan)
            rt.append(np.nan)

            if task == 'congruentstroop':
                condition.append('congruent')
            elif task == 'incongruentstroop':
                condition.append('incongruent')
            elif task == 'mixedstroop':
                if new_evs[event, 2] in {1, 2, 3, 4}:
                    condition.append('congruent')
                elif new_evs[event, 2] in {101, 102, 103, 104}:
                    condition.append('incongruent')

            # proceed to next stimulus
            continue

    # skip other events
    else:
        continue

###############################################################################
# 4) Set descriptive event names for extraction of epochs

# cue events
stroop_event_id = {'correct green': 11,
                   'correct red': 12,
                   'correct yellow': 13,
                   'correct blue': 14,

                   'incorrect green': 31,
                   'incorrect red': 32,
                   'incorrect yellow': 33,
                   'incorrect blue': 34
                   }


# ##############################################################################
# # 5) Create metadata structure to be added to the epochs

# # only keep word events
word_events = new_evs[np.where((new_evs[:, 2] == 11) |
                               (new_evs[:, 2] == 12) |
                               (new_evs[:, 2] == 13) |
                               (new_evs[:, 2] == 14) |

                               (new_evs[:, 2] == 31) |
                               (new_evs[:, 2] == 32) |
                               (new_evs[:, 2] == 33) |
                               (new_evs[:, 2] == 34))]

# # reversed event_id dict
# cue_event_id_rev = {val: key for key, val in cue_event_id.items()}
# probe_event_id_rev = {val: key for key, val in probe_event_id.items()}

# check if events shape match
# if cue_events.shape[0] != probe_events.shape[0]:
#     cue_events = np.delete(cue_events, broken, 0)
#
# # create list with reactions based on cue and probe event ids
# same_stim, reaction_cues, reaction_probes, cues, probes = [], [], [], [], []
# for cue, probe in zip(cue_events[:, 2], probe_events[:, 2]):
#     response, cue = cue_event_id_rev[cue].split(' ')
#     reaction_cues.append(response)
#     # save cue
#     cues.append(cue)
#
#     # save response
#     response, probe = probe_event_id_rev[probe].split(' ')
#     reaction_probes.append(response)
#
#     # check if same type of combination was shown in the previous trail
#     if len(probes):
#         stim = same_stim[-1]
#         if probe == probes[-1] \
#                 and response == 'Correct' \
#                 and reaction_probes[-2] == 'Correct':
#             stim += 1
#             same_stim.append(stim)
#         else:
#             same_stim.append(0)
#     else:
#         stim = 0
#         same_stim.append(0)
#
#     # save probe
#     probes.append(probe)

# create data frame with epochs metadata
metadata = {'trial': good_trials,
            'color': color,
            'rt': rt,
            'reaction': reaction,
            'condition': condition
            }
metadata = pd.DataFrame(metadata)

# set run info
run = ''
if task == 'congruentstroop':
    run = 'congruent stroop'
elif task == 'incongruentstroop':
    run = 'incongruent stroop'
elif task == 'mixedstroop':
    run = 'mixed stroop'
metadata['run'] = run


# # save RT measures for later analyses
# rt_data = metadata.copy()
# rt_data = rt_data.assign(subject=subject)
#
# # save to disk
# rt_data.to_csv(op.join(fname.rt, 'sub-%s-rt.tsv' % subject),
#                sep='\t',
#                index=False)

###############################################################################
# 6) Extract the epochs

# rejection threshold
reject = dict(eeg=100e-6)
decim = 1

if raw.info['sfreq'] == 1000.0:
    decim = 8


# extract cue epochs
stroop_epochs = Epochs(raw, word_events, stroop_event_id,
                       metadata=metadata,
                       on_missing='ignore',
                       tmin=-3.0,
                       tmax=2.0,
                       baseline=None,
                       preload=True,
                       reject_by_annotation=True,
                       reject=reject,
                       decim=decim
                       )

###############################################################################
# 8) Save epochs

# output path for cues
epochs_output_path = fname.output(subject=subject,
                                  task=task,
                                  processing_step='epochs',
                                  file_type='epo.fif')

# resample and save cue epochs to disk
stroop_epochs.save(epochs_output_path, overwrite=True)

###############################################################################
# # 9) Create HTML report
# epochs_summary = '<p>Cue epochs extracted: <br>' \
#                  'A: %s <br>' \
#                  'B: %s <br>' \
#                  '<p>Probe epochs extracted: <br>' \
#                  'AX: %s <br>' \
#                  'AY: %s <br>' \
#                  'BX: %s <br>' \
#                  'BY: %s <br>' \
#                  % (
#                      len(cue_epochs['Correct A']),
#                      len(cue_epochs['Correct B']),
#                      len(probe_epochs['Correct AX']),
#                      len(probe_epochs['Correct AY']),
#                      len(probe_epochs['Correct BX']),
#                      len(probe_epochs['Correct BY'])
#                     )
#
# with open_report(fname.report(subject=subject)[0]) as report:
#     report.add_htmls_to_section(htmls=epochs_summary,
#                                 captions='Epochs summary',
#                                 section='Epochs',
#                                 replace=True)
#     report.save(fname.report(subject=subject)[1], overwrite=True,
#                 open_browser=False)

###############################################################################
# 10) Save time bins for further analysis
#
# # extract corrects
# correct_epochs = cue_epochs['Correct A', 'Correct B']
# # apply baseline correction
# correct_epochs.apply_baseline((-0.300, -0.050))
#
# # epochs to pandas dataframe
# df = correct_epochs.to_data_frame(long_format=True)
#
# # get time roi N170
# n170 = df[((df["time"] >= 150) & (df["time"] <= 250))
#           & ((df["channel"] == 'PO8') |
#              (df["channel"] == 'PO7') |
#              (df["channel"] == 'FCz'))]
# n170 = n170.assign(subject=subject)
#
# # get time roi LPC
# LPC = df[((df["time"] >= 400) & (df["time"] <= 750))
#          & (df["channel"] == 'Pz')]
# LPC = LPC.assign(subject=subject)
#
# # get time roi CNV
# CNV = df[((df["time"] >= 1000) & (df["time"] <= 1750))
#          & (df["channel"] == 'C1')]
# CNV = CNV.assign(subject=subject)
#
# # export to .tsv files
# n170.to_csv(op.join(fname.rois,
#                     'sub-%s-n170.tsv' % subject),
#             sep='\t',
#             index=False)
# LPC.to_csv(op.join(fname.rois,
#                    'sub-%s-LPC.tsv' % subject),
#            sep='\t',
#            index=False)
#
# CNV.to_csv(op.join(fname.rois,
#                    'sub-%s-cnv.tsv' % subject),
#            sep='\t',
#            index=False)
