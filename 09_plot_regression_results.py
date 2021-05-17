"""
=====================================
Compute T-map for effect of condition
=====================================

Mass-univariate analysis of cue evoked activity.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

import pandas as pd

import patsy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne.stats.cluster_level import _setup_adjacency, _find_clusters
from mne.channels import find_ch_adjacency
from mne.evoked import EvokedArray
from mne import read_epochs, grand_average

from config import fname

# subjects
subjects = np.load(fname.results + '/analysed_subjects.npy')

# get subject id and group
ids = pd.read_csv('../data_bids/subject_data/subject_ids.tsv',
                  sep='\t',
                  header=0)
ids = ids.loc[ids['subject_id'].isin(subjects)]
ids = ids[['subject_id', 'group_id']].sort_values(by='subject_id')

# also load individual beta coefficients
betas = np.load(fname.results + '/subj_betas_condition_m250_robust.npy')
r2 = np.load(fname.results + '/subj_r2_condition_m250_robust.npy')

###############################################################################
# 1) import epochs to use as template

# baseline for epochs
baseline = (-0.35, -0.05)

# import the output from previous processing step
input_file = fname.output(subject=int(subjects[0]),
                          task='congruentstroop',
                          processing_step='epochs',
                          file_type='epo.fif')
stroop_epo = read_epochs(input_file, preload=True)
stroop_epo = stroop_epo['reaction == "correct"']
stroop_epo_nb = stroop_epo.copy().crop(tmin=-0.25, tmax=2.0,
                                       include_tmax=True)
stroop_epo = stroop_epo.apply_baseline(baseline).crop(tmin=-0.35)

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.).
epochs_info = stroop_epo_nb.info
n_channels = len(epochs_info['ch_names'])
n_times = len(stroop_epo_nb.times)
times = stroop_epo_nb.times
tmin = stroop_epo_nb.tmin

# placeholder for results
betas_evoked = dict()
r2_evoked = dict()

# ###############################################################################
# 2) loop through subjects and extract betas
for n_subj, subj in enumerate(subjects):
    subj_beta = betas[n_subj, :]
    subj_beta = subj_beta.reshape((n_channels, n_times))
    betas_evoked[str(subj)] = EvokedArray(subj_beta, epochs_info, tmin)

    subj_r2 = r2[n_subj, :]
    subj_r2 = subj_r2.reshape((n_channels, n_times))
    r2_evoked[str(subj)] = EvokedArray(subj_r2, epochs_info, tmin)

effect_of_condition = grand_average([betas_evoked[str(subj)] for subj in
                                     subjects])
cue_r2 = grand_average([r2_evoked[str(subj)] for subj in subjects])

###############################################################################
# 3) Plot beta weights for the effect of condition

# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-5.5, 5.5]),
               xlim=[-.25, 2.0])

# times to plot
ttp = [0.17, 0.21, 0.30, 0.5, 0.60, 0.70, 1.00]

# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='ms',
                    vmin=6, vmax=-6,
                    average=0.05,
                    extrapolate='head',
                    outlines='head')

# create plot
title = 'Regression coefficients (Incong. - Cong., 34 EEG channels)'
fig = effect_of_condition.plot_joint(ttp,
                                     ts_args=ts_args,
                                     topomap_args=topomap_args,
                                     title=title,
                                     show=False)
fig.axes[-1].texts[0]._fontproperties._size = 12.0  # noqa
fig.axes[-1].texts[0]._fontproperties._weight = 'bold'  # noqa
fig.axes[0].set_xticks(list(np.arange(-0.25, 2.05, 0.25)), minor=False)
fig.axes[0].set_yticks(list(np.arange(-5.0, 5.5, 2.5)), minor=False)
fig.axes[0].set_xticklabels(list(np.arange(-250, 2050, 250)))
fig.axes[0].set_xlabel('Time (ms)')
fig.axes[0].axhline(y=0.0, xmin=-0.5, xmax=2.0,
                    color='black', linestyle='dashed', linewidth=0.8)
fig.axes[0].axvline(x=0.0, ymin=-6.0, ymax=6.0,
                    color='black', linestyle='dashed', linewidth=0.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(-5.0, 5.0)
fig.axes[0].spines['bottom'].set_bounds(-0.25, 2.0)
fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1.15, h * 1.15)
fig_name = fname.figures + '/Evoked_average_betas_condtion.pdf'
fig.savefig(fig_name, dpi=300)

###############################################################################
# 4) Compute F values for effect of condition

# force betas array to have 3 dimensions,
# i.e., (subjects, level 1 predictors, time samples)
if len(betas.shape) == 2:
    betas = betas[:, np.newaxis, :]
    n_contrasts = 1
else:
    n_contrasts = betas.shape[1]

n = betas.shape[0]

# compute error matrix for the full factorial model
n_group_levels = len(np.unique(ids[['group_id']].to_numpy()))
group_design = pd.get_dummies(ids[['group_id']], drop_first=False)
group_design = group_design.assign(Intercept=1).to_numpy()
R = np.eye(n) - group_design @ np.linalg.pinv(group_design)

E = np.zeros((n_contrasts, betas.shape[-1]))
for t in range(betas.shape[-1]):
    print(t)
    E[:, t] = betas[:, :, t].T @ R @ betas[:, :, t]

# n levels - 1
df = n_contrasts
dfe = (n-(n_contrasts+1)) - (n_group_levels - 1)

mean_betas = betas.mean(axis=0)

# np.sum(X.to_numpy()[:, :-1].sum(axis=0)-1)
ve = np.sum(group_design[:, :-1].sum(axis=0)-1)

Spl = E / ve

F = np.zeros((n_times * n_channels))

for t in range(mean_betas.shape[-1]):
    print(t)
    m_at_t = mean_betas[:, t]
    if len(Spl[:, t].shape) == 1:
        cov_inv = 1 / Spl[:, t]
    else:
        cov_inv = np.linalg.inv(Spl[:, :, t])
    if len(cov_inv.shape) == 1:
        T2 = n * m_at_t * cov_inv * m_at_t  # Hotelings T-Square at time t
    else:
        T2 = n * m_at_t @ cov_inv @ m_at_t

    F[t] = (dfe / (ve * df)) * T2

F = F.reshape((n_channels, n_times))
F_evoked = EvokedArray(F, epochs_info, tmin)

font = 'Mukta'  # noqa
fig, ax = plt.subplots(figsize=(10, 10))
axes = [plt.subplot2grid((10, 6), (0, 0), rowspan=10, colspan=4),
        plt.subplot2grid((10, 6), (2, 5), rowspan=6, colspan=1),
        ]

F_evoked.plot_image(cmap='magma',
                    mask=F > 10.0,
                    axes=axes[0],
                    clim=dict(eeg=[0, F.max()]),
                    unit=False,
                    scalings=dict(eeg=1),
                    colorbar=False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_bounds(0, 33)
axes[0].spines['left'].set_linewidth(1.5)
axes[0].spines['bottom'].set_bounds(-0.250, 2.0)
axes[0].spines['bottom'].set_linewidth(1.5)

axes[0].set_ylabel('EEG sensors', labelpad=10.0, fontsize=10.5)
axes[0].set_yticks(np.arange(0, 34, 1))
axes[0].set_yticklabels(F_evoked.ch_names, fontname=font)

# add line marking stimulus presentation
axes[0].axvline(x=0, ymin=0, ymax=33,
                color='black', linestyle='dashed', linewidth=1.0)

colormap = cm.get_cmap('magma')
orientation = 'vertical'
norm = Normalize(vmin=0, vmax=F.max())
cbar = ColorbarBase(axes[-1],
                    cmap=colormap,
                    ticks=np.arange(0, F.max(), 5),
                    norm=norm,
                    orientation=orientation)
cbar.outline.set_visible(False)
cbar.ax.set_frame_on(True)
cbar.ax.tick_params(labelsize=9)
cbar.set_label(label=r'Main Effect Incong. vs Cong. ($F-value$)', font=font,
               size=12)
for key in ('left', 'top',
            'bottom' if orientation == 'vertical' else 'right'):
    cbar.ax.spines[key].set_visible(False)
for tk in cbar.ax.yaxis.get_ticklabels():
    tk.set_family(font)
fig.subplots_adjust(
    wspace=0.1, hspace=0.5)
fig_name = fname.figures + '/F_test_condition_diff.pdf'
fig.savefig(fig_name, dpi=300)


# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               unit=False,
               ylim=dict(eeg=[-0.005, F.max()+1]),
               xlim=[-0.25, 1.0])

# times to plot
ttp = [0.215, 0.30, 0.40, 0.50, 0.60, 0.75]

# arguments fot the topographical maps
topomap_args = dict(cmap='magma_r',
                    scalings=dict(eeg=1),
                    sensors=False,
                    time_unit='ms',
                    vmin=0.0, vmax=F.max(),
                    average=0.05,
                    extrapolate='head',
                    outlines='head')

title = 'Main effect incong. vs cong.'
fig = F_evoked.plot_joint(ttp,
                          ts_args=ts_args,
                          topomap_args=topomap_args,
                          title=title,
                          show=False)
fig.axes[-1].texts[0]._fontproperties._size = 12.0  # noqa
fig.axes[-1].texts[0]._fontproperties._weight = 'bold'  # noqa
fig.axes[0].set_xticks(list(np.arange(-0.25, 1.05, 0.25)), minor=False)
fig.axes[0].set_xticklabels(list(np.arange(-250, 1005, 250)))
fig.axes[0].set_xlabel('Time (ms)')
fig.axes[0].set_yticks(list(np.arange(0.0, F.max()+2, 4)), minor=False)
fig.axes[0].axvline(x=0.0, ymin=0.0, ymax=F.max()+2,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(0.0, F.max()+1)
fig.axes[0].spines['bottom'].set_bounds(-0.25, 1.0)
fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
fig.axes[0].set_ylabel('F-value')
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1.15, h * 1.15)
fig_name = fname.figures + '/Evoked_F_condition.pdf'
fig.savefig(fig_name, dpi=300)


# # test
# data = np.zeros((20, 11, 4))
# data[:, :, 0] = np.asarray([1, 2, 3, 8, 5, 2, 4, 8, 7, 2, 4])
# data[:, :, 1] = np.asarray([1, 5, 7, 9, 4, 2, 6, 4, 8, 3, 5])
# data[:, :, 2] = np.asarray([4, 6, 8, 4, 1, 8, 0, 1, 4, 6, 2])
# data[:, :, 3] = np.asarray([8, 9, 6, 8, 5, 8, 2, 7, 5, 9, 4])
#
# R2 = np.eye(11) - X @ np.linalg.pinv(X)
# E2 = np.zeros((20, 4, 4))
# for t in range(E2.shape[0]):
#     print(t)
#     E2[t, :, :] = data[t, :, :].T @ R2 @ data[t, :, :]
#
# df1 = 3
# df2 = 7
# yp = data.mean(axis=1)
# ve2 = 9
# Spl2 = E2 / ve2
#
# FF = np.zeros((20))
# for t in range(FF.shape[0]):
#     print(t)
#     m_at_t = yp[t, :]
#     T2 = n * m_at_t @ cov_inv @ m_at_t  # Hotelings T-Squareb at time = t
#
#
#
# data[0, :, :].T @ R2 @ data[0, :, :]
#
