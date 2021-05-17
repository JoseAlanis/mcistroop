"""
=============================================
Fit single subject linear model to cue epochs
=============================================

Mass-univariate analysis of cue evoked activity.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os

import numpy as np
import patsy

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mne import read_epochs, concatenate_epochs
from mne.decoding import Vectorizer, get_coef

# all parameters are defined in config.py
from config import fname, LoggingFormat

# baseline to be applied
baseline = (-0.35, -0.05)

# subjects to analyse
subjects = np.arange(1, 63)

# dicts for storing individual sets of epochs/ERPs
block_stroop = dict()

###############################################################################
# 1) loop through subjects and extract condition specific epochs
for subj in subjects:

    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Loading epochs for subject %s' % subj +
          LoggingFormat.END)

    # import the output from previous processing step
    input_cong = fname.output(subject=subj,
                              task='congruentstroop',
                              processing_step='epochs',
                              file_type='epo.fif')

    # import the output from previous processing step
    input_incong = fname.output(subject=subj,
                                task='incongruentstroop',
                                processing_step='epochs',
                                file_type='epo.fif')

    # check if file exists, otherwise terminate the script
    if not os.path.isfile(input_cong) or not os.path.isfile(input_incong):
        continue

    # load epochs and only keep correct reactions
    cong_epo = read_epochs(input_cong, preload=True)
    cong_epo = cong_epo['reaction == "correct"']
    cong_epo = cong_epo.apply_baseline(baseline).crop(tmin=-0.5)

    # load epochs and only keep correct reactions
    incong_epo = read_epochs(input_incong, preload=True)
    incong_epo = incong_epo['reaction == "correct"']
    incong_epo = incong_epo.apply_baseline(baseline).crop(tmin=-0.5)

    # combine epochs from both stroop blocks to a single structure
    block_epochs = concatenate_epochs([cong_epo, incong_epo])

    # store concatenated epochs in a dict for further analysis
    block_stroop['%s' % subj] = block_epochs

###############################################################################
# 2) linear model parameters
# use first subject as generic information template for results
generic = block_stroop['%s' % subjects[0]].copy()

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.).
epochs_info = generic.info

# only use times > -0.25
times_to_use = (generic.times >= -0.25) & (generic.times <= 2.0)
times = generic.times[times_to_use]
n_times = len(times)
n_channels = len(epochs_info['ch_names'])

# subjects
subjects = list(block_stroop.keys())
analysed_subjects = np.asarray([key for key in block_stroop.keys()])
np.save(fname.results + '/analysed_subjects.npy', analysed_subjects)

# independent variables to be used in the analysis (i.e., predictors)
predictors = ['condition']

# number of predictors
n_predictors = len(predictors)

###############################################################################
# 3) initialise place holders for the storage of results
betas = np.zeros((len(block_stroop.values()),
                  n_channels * n_times))

r_squared = np.zeros((len(block_stroop.values()),
                      n_channels * n_times))

###############################################################################
# 4) Fit linear model for each subject
for subj_ind, subj in enumerate(block_stroop):
    print(subj_ind, subj)

    # 4.1) create subject design matrix using epochs metadata
    metadata = block_stroop[subj].metadata.copy()

    # only keep predictor columns
    design = metadata[predictors]

    # create design matrix
    design = patsy.dmatrix("condition", design, return_type='dataframe')
    design = design[['condition[T.incongruent]']]

    # 4.2) vectorise channel data for linear regression
    # data to be analysed
    dat = block_stroop[subj].get_data()
    dat = dat[:, :, times_to_use]
    Y = Vectorizer().fit_transform(dat)

    # 4.3) fit linear model with sklearn's LinearRegression
    weights = compute_sample_weight(class_weight='balanced',
                                    y=metadata.condition.to_numpy())
    linear_model = LinearRegression(n_jobs=2, fit_intercept=True)
    linear_model.fit(design, Y, sample_weight=weights)

    # 4.4) extract the resulting coefficients (i.e., betas)
    # extract betas
    coefs = get_coef(linear_model, 'coef_')
    inter = linear_model.intercept_

    # 4.5) extract model r_squared
    r2 = r2_score(Y, linear_model.predict(design),
                  multioutput='raw_values')
    # save model R-squared
    r_squared[subj_ind, :] = r2

    # save results
    for pred_i, predictor in enumerate(design.columns):
        print(pred_i, predictor)
        if 'condition' in predictor:
            # extract cue beats
            betas[subj_ind, :] = coefs[:, pred_i]
        # elif 'Intercept' in predictor:
        #     continue

###############################################################################
# 5) Save subject-level results to disk
np.save(fname.results + '/subj_betas_condition_m250_robust.npy', betas)
np.save(fname.results + '/subj_r2_condition_m250_robust.npy', r_squared)
