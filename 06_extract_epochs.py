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
# fixed names for events
new_ids = dict()
if task == 'congruentstroop':
    new_ids = {
        # fix cross
        'fix cross': 100,

        # start of tasks
        'start S/C': 501,

        'C/C/G': 101,
        'C/C/R': 201,
        'C/C/Y': 301,
        'C/C/B': 401,

        # responses
        'G': 1,
        'R': 2,
        'Y': 3,
        'B': 4}
elif task == 'incongruentstroop':
    new_ids = {
        # fix cross
        'fix cross': 100,

        # start of tasks
        'start S/I': 502,

        'I/C/G': 102,
        'I/C/R': 202,
        'I/C/Y': 302,
        'I/C/B': 402,

        # responses
        'G': 1,
        'R': 2,
        'Y': 3,
        'B': 4}
elif task == 'mixedstroop':
    new_ids = {
        # fix cross
        'fix cross': 100,

        # start of tasks
        'start S/I': 502,

        'M/C/G': 103,
        'M/I/G': 104,

        'M/C/R': 203,
        'M/I/R': 204,

        'M/C/Y': 303,
        'M/I/Y': 304,

        'M/C/B': 403,
        'M/I/B': 404,

        # responses
        'G': 1,
        'R': 2,
        'Y': 3,
        'B': 4}

# extract events
events = events_from_annotations(raw, event_id=new_ids, regexp=None)

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
# block = []
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
        if new_evs[event + 1, 2] in \
                {101, 102, 103, 104,
                 201, 202, 203, 204,
                 301, 302, 303, 304,
                 401, 402, 403, 404}:
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
    elif new_evs[event, 2] in \
            {101, 102, 103, 104,
             201, 202, 203, 204,
             301, 302, 303, 304,
             401, 402, 403, 404}:

        # -- if event is the last one --
        if new_evs[event, 0] == new_evs[-1, 0]:

            # store condition information
            if new_evs[event, 2] in {101, 201, 301, 401}:
                condition.append('congruent')
            elif new_evs[event, 2] in {102, 202, 302, 402}:
                condition.append('incongruent')
            elif new_evs[event, 2] in {103, 203, 303, 403}:
                condition.append('congruent')
            elif new_evs[event, 2] in {104, 204, 304, 404}:
                condition.append('incongruent')

            # add event as missed reaction
            miss.append(trial)
            reaction.append(np.nan)
            rt.append(np.nan)
            trial += 1

            # proceed to next stimulus as no reaction was provided
            continue

        # -- if event is followed by a reaction but trial is invalid --
        elif new_evs[event + 1, 2] in {1, 2, 3, 4} and not valid_trial:

            # store condition information and reaction as missing
            if new_evs[event, 2] in {101, 201, 301, 401}:
                condition.append('congruent')
            elif new_evs[event, 2] in {102, 202, 302, 402}:
                condition.append('incongruent')
            elif new_evs[event, 2] in {103, 203, 303, 403}:
                condition.append('congruent')
            elif new_evs[event, 2] in {104, 204, 304, 404}:
                condition.append('incongruent')
            miss.append(trial)
            reaction.append(np.nan)
            rt.append(np.nan)
            trial += 1

            # proceed to next stimulus
            continue

        # -- if event is a word but no reaction followed
        elif new_evs[event + 1, 2] not in {1, 2, 3, 4}:

            # store condition information and reaction as missing
            if new_evs[event, 2] in {101, 201, 301, 401}:
                condition.append('congruent')
            elif new_evs[event, 2] in {102, 202, 302, 402}:
                condition.append('incongruent')
            elif new_evs[event, 2] in {103, 203, 303, 403}:
                condition.append('congruent')
            elif new_evs[event, 2] in {104, 204, 304, 404}:
                condition.append('incongruent')
            miss.append(trial)
            reaction.append(np.nan)
            rt.append(np.nan)
            trial += 1

            # proceed to next stimulus
            continue

        # -- if event is followed by a reaction and trial is valid --
        elif new_evs[event + 1, 2] in {1, 2, 3, 4} and valid_trial:

            # add condition info
            col = str(new_evs[event, 2])[0]
            cond = str(new_evs[event, 2])[-1]
            if cond == '1':
                condition.append('congruent')
            elif cond == '2':
                condition.append('incongruent')
            elif cond == '3':
                condition.append('congruent')
            elif cond == '4':
                condition.append('incongruent')

            # --- if color green ---
            if new_evs[event, 2] in {101, 102, 103, 104}:
                # add color and condition
                color.append('green')

            # --- if color red ---
            elif new_evs[event, 2] in {201, 202, 203, 204}:
                # add color and condition
                color.append('red')

            # --- if color yellow ---
            elif new_evs[event, 2] in {301, 302, 303, 304}:
                # add color and condition
                color.append('yellow')

            # --- if color blue ---
            elif new_evs[event, 2] in {401, 402, 403, 404}:
                # add color and condition
                color.append('blue')

            # compute reaction correctness
            reac = str(new_evs[event + 1, 2])
            if col == reac:
                reaction.append('correct')
                new_evs[event + 1, 2] = new_evs[event + 1, 2] + 1000
                new_evs[event, 2] = new_evs[event, 2] + 1000
            else:
                new_evs[event + 1, 2] = new_evs[event + 1, 2] + 2000
                new_evs[event, 2] = new_evs[event, 2] + 2000
                reaction.append('incorrect')

            # compute reaction time
            r_time = (new_evs[event + 1, 0] - new_evs[event, 0]) / sfreq
            rt.append(r_time)

            # store trial id as good trial
            good_trials.append(trial)
            trial += 1
            continue

    # skip other events
    else:
        continue

###############################################################################
# 4) Set descriptive event names for extraction of epochs
stroop_event_id = dict()
if task == 'congruentstroop':
    # cue events
    stroop_event_id = {'correct green congruent': 1101,
                       'correct red congruent': 1201,
                       'correct yellow congruent': 1301,
                       'correct blue congruent': 1401,

                       'incorrect green congruent': 2101,
                       'incorrect red congruent': 2201,
                       'incorrect yellow congruent': 2301,
                       'incorrect blue congruent': 2401
                       }
elif task == 'incongruentstroop':
    stroop_event_id = {'correct green incongruent': 1102,
                       'correct red incongruent': 1202,
                       'correct yellow incongruent': 1302,
                       'correct blue incongruent': 1402,

                       'incorrect green incongruent': 2102,
                       'incorrect red incongruent': 2202,
                       'incorrect yellow incongruent': 2302,
                       'incorrect blue incongruent': 2402
                       }
elif task == 'mixedstroop':
    stroop_event_id = {'correct green congruent': 1103,
                       'correct green incongruent': 1104,

                       'correct red congruent': 1203,
                       'correct red incongruent': 1204,

                       'correct yellow congruent': 1303,
                       'correct yellow incongruent': 1304,

                       'correct blue congruent': 1403,
                       'correct blue incongruent': 1404,

                       'incorrect green congruent': 2103,
                       'incorrect green incongruent': 2104,

                       'incorrect red congruent': 2204,
                       'incorrect red incongruent': 2204,

                       'incorrect yellow congruent': 2303,
                       'incorrect yellow incongruent': 2304,

                       'incorrect blue congruent': 2403,
                       'incorrect blue incongruent': 2404
                       }


# ##############################################################################
# # 5) Create metadata structure to be added to the epochs

word_events = dict()
if task == 'congruentstroop':
    # only keep word events
    word_events = new_evs[np.where((new_evs[:, 2] == 1101) |
                                   (new_evs[:, 2] == 1201) |
                                   (new_evs[:, 2] == 1301) |
                                   (new_evs[:, 2] == 1401) |

                                   (new_evs[:, 2] == 2101) |
                                   (new_evs[:, 2] == 2201) |
                                   (new_evs[:, 2] == 2301) |
                                   (new_evs[:, 2] == 2401))]
elif task == 'incongruentstroop':
    # only keep word events
    word_events = new_evs[np.where((new_evs[:, 2] == 1102) |
                                   (new_evs[:, 2] == 1202) |
                                   (new_evs[:, 2] == 1302) |
                                   (new_evs[:, 2] == 1402) |

                                   (new_evs[:, 2] == 2102) |
                                   (new_evs[:, 2] == 2202) |
                                   (new_evs[:, 2] == 2302) |
                                   (new_evs[:, 2] == 2402))]
elif task == 'mixedstroop':
    # only keep word events
    word_events = new_evs[np.where((new_evs[:, 2] == 1103) |
                                   (new_evs[:, 2] == 1203) |
                                   (new_evs[:, 2] == 1303) |
                                   (new_evs[:, 2] == 1403) |
                                   (new_evs[:, 2] == 1104) |
                                   (new_evs[:, 2] == 1204) |
                                   (new_evs[:, 2] == 1304) |
                                   (new_evs[:, 2] == 1404) |

                                   (new_evs[:, 2] == 2103) |
                                   (new_evs[:, 2] == 2203) |
                                   (new_evs[:, 2] == 2303) |
                                   (new_evs[:, 2] == 2403) |
                                   (new_evs[:, 2] == 2104) |
                                   (new_evs[:, 2] == 2204) |
                                   (new_evs[:, 2] == 2304) |
                                   (new_evs[:, 2] == 2404))]

# create data frame with epochs metadata
metadata = {'trial': good_trials,
            'color': color,
            'rt': np.delete(rt, miss),
            'reaction': np.delete(reaction, miss),
            'condition': np.delete(condition, miss)
            }
metadata = pd.DataFrame(metadata)

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
reject = dict(eeg=250e-6)
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
