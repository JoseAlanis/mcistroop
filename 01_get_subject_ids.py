"""
=================================
Create unique subject identifiers
=================================

Subject IDs (and session names) are inferred based on the EEG files names.


Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os
import glob
import re

import pandas as pd

from config import input_path, output_bids

# naming pattern
# pattern = '\\d+[_-]+\\d+[_-]+[a-zA-Z].vhdr'

# get subject ids
subjects = pd.read_csv('../subjects_list.csv', sep=';', header=0)

##############################################################################
# 1) extract list of files in directory
files = glob.glob(os.path.join(input_path, '*.vhdr'))
files.sort()
files = [os.path.basename(file) for file in files]

# only keep files that are named as expected (roughly)
# files = [file for file in files if re.search(pattern, file)]
# remove ambiguous
# files = [file for file in files if file != '21_08_M.vhdr']

##############################################################################
# 2) loop through files and create unique subject and session IDs based on the
# file naming pattern

sub_ids = []
eeg_ids = []
group_ids = []
for n_file, file in enumerate(files):

    print(n_file, file)

    # extract file name components
    if file == '31-07.vhdr':
        d = '31'
        m = '07'
    else:
        id_0, id_1 = \
            [int(re.split('[_-]', file)[i]) for i in [0, 1]]
        # make sure they are in the correct format
        d = str(id_0).rjust(2, '0')
        m = str(id_1).rjust(2, '0')

    # create measurement date
    meas_date = '2020-%s-%s' % (m, d)

    # extract session identifier
    # (some files have weird names)
    if file == '03-08-e-a.vhdr':
        sess = 'E'
    elif file == '06_08_M_not included.vhdr':
        sess = 'M'
    elif file == '14_08_M.D without baselin.vmrk.vhdr':
        sess = 'M'
    elif file == '16-09-M-badIM.vhdr':
        sess = 'M'
    elif file == '18_8_M _exclude.vhdr':
        sess = 'M'
    elif file == '31-07.vhdr':
        sess = 'M'
    elif file == '31_08_M not G.vhdr':
        sess = 'M'
    else:
        sess = re.split('[_-]', file)[-1].split('.')[0]

    # get subject information
    subj_row = subjects.loc[(subjects['Measurement_Date'] == meas_date) &
                            (subjects['Measurement_Time'] == sess)]

    # put file identifiers in a "correct" (more readable) format,
    # while keeping the logic behind the original names
    eeg_id = '%s-%s-%s' % (d, m, sess)
    eeg_ids.append(eeg_id)

    sub_id = subj_row['Subject'].values[0]
    sub_ids.append(sub_id)

    group_id = subj_row['Group'].values[0]
    group_ids.append(group_id)


##############################################################################
# 3) put everything together in a pandas dataframe and export it to a .tsv file
subject_identifiers = {'file_name': files,
                       'eeg_id': eeg_ids,
                       'subject_id': sub_ids,
                       'group_id': group_ids}
subject_identifiers = pd.DataFrame(subject_identifiers)

# check if target dir exists
subject_data_path = os.path.join(output_bids, 'subject_data')
os.makedirs(subject_data_path, exist_ok=True)

# save subject data
subject_identifiers.to_csv(os.path.join(subject_data_path, 'subject_ids.tsv'),
                           sep='\t',
                           index=False)
