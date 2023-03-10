#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

@author: Duan Charles
@time:  12:45 下午

'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from scipy.stats import entropy

# %%


# ground truth song attr and songs
test_dat_attr = pickle.load(open('../../results/test/y_test_dat_attr.pkl', 'rb'))
test_dat_p_attr, test_dat_d_attr, test_dat_r_attr = test_dat_attr
test_dat_songs = np.load('../../results/test/test_dat_songs.npy')

# songs generated using C-Hybrid-GAN
test_c_hybrid_gan_attr = pickle.load(
    open('../../results/c_hybrid_gan/generated/c_hybrid_gan_y_test_gen_attr.pkl', 'rb'))
test_c_hybrid_gan_p_attr, test_c_hybrid_gan_d_attr, test_c_hybrid_gan_r_attr = test_c_hybrid_gan_attr
test_c_hybrid_gan_songs = np.load('../../results/c_hybrid_gan/generated/c_hybrid_gan_test_gen_songs.npy')

# songs generated using C-Hybrid-MLE
test_c_hybrid_mle_attr = pickle.load(
    open('../../results/c_hybrid_gan/generated/c_hybrid_mle_y_test_gen_attr.pkl', 'rb'))
test_c_hybrid_mle_p_attr, test_c_hybrid_mle_d_attr, test_c_hybrid_mle_r_attr = test_c_hybrid_mle_attr
test_c_hybrid_mle_songs = np.load('../../results/c_hybrid_gan/generated/c_hybrid_mle_test_gen_songs.npy')

# songs generated using C-LSTM-GAN
test_c_lstm_gan_p_attr = np.load('../../results/baseline/c_lstm_gan_y_test_p_attr.npy')

# %% md

### Dataset Distribution of Music Attributes (Fig. 5)

# %%


SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

# %%


plt.figure(figsize=(20, 6))

plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

plt.subplot(131)
p, num_p = np.unique(np.ravel(test_dat_p_attr), return_counts=True)
plt.bar(p, num_p / sum(num_p))
plt.xlabel('MIDI Number')
plt.ylabel('Occurence Probability')

plt.subplot(132)
d, num_d = np.unique(np.ravel(test_dat_d_attr), return_counts=True)
d, num_d = np.append(d, [32.]), np.append(num_d, [0])
plt.bar(['1/4', '1/2', '3/4', '1', '3/2', '2', '3', '4', '6', '8', '16', '32'], num_d / sum(num_d), color='tab:green')
plt.xlabel('Note Duration')
plt.ylabel('Occurence Probability')

plt.subplot(133)
r, num_r = np.unique(np.ravel(test_dat_r_attr), return_counts=True)
plt.bar(['0', '1', '2', '4', '8', '16', '32'], num_r / sum(num_r), color='tab:orange')
plt.xlabel('Rest Duration')
plt.ylabel('Occurence Probability')

plt.tight_layout()
plt.savefig('./figures/attr.png', dpi=400)


# %% md

### Distribution of Transitions (Fig. 9)

# %%

def transition_util(p_attr, steps=8):
    transition = p_attr[:, 1:] - p_attr[:, :-1]
    trans, num_trans = np.unique(np.ravel(transition), return_counts=True)
    center = np.argmax(num_trans)
    labels = [str(int(i)) for i in trans[(center - steps):(center + steps)]]
    return labels, num_trans[(center - steps):(center + steps)]


# %%


plt.figure(figsize=(22, 11))
rcParams['figure.dpi'] = 400

plt.rc('axes', titlesize=22, labelsize=22)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels

x1, y1 = transition_util(test_dat_p_attr)

plt.subplot(221)
plt.bar(x1, y1)
plt.title('Melodies in testing set')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x2, y2 = transition_util(test_c_hybrid_gan_p_attr)

plt.subplot(222)
plt.bar(x2, y2)
plt.title(f'Melodies generated by C-Hybrid-GAN\nKL Divergence = {round(entropy(y1, y2), 4)}')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x3, y3 = transition_util(test_c_hybrid_mle_p_attr)

plt.subplot(223)
plt.bar(x3, y3)
plt.title(f'Melodies generated by C-Hybrid-MLE\nKL Divergence = {round(entropy(y1, y3), 4)}')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x4, y4 = transition_util(test_c_lstm_gan_p_attr)

plt.subplot(224)
plt.bar(x4, y4)
plt.title(f'Melodies generated by C-LSTM-GAN\nKL Divergence = {round(entropy(y1, y4), 4)}')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

plt.tight_layout()
plt.savefig('./figures/transitions_with_kl.png')

# %%


plt.figure(figsize=(22, 11))
rcParams['figure.dpi'] = 400

plt.rc('axes', titlesize=22, labelsize=22)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels

x1, y1 = transition_util(test_dat_p_attr)

plt.subplot(221)
plt.bar(x1, y1)
plt.title('Melodies in testing set')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x2, y2 = transition_util(test_c_hybrid_gan_p_attr)

plt.subplot(222)
plt.bar(x2, y2)
plt.title(f'Melodies generated by C-Hybrid-GAN')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x3, y3 = transition_util(test_c_hybrid_mle_p_attr)

plt.subplot(223)
plt.bar(x3, y3)
plt.title(f'Melodies generated by C-Hybrid-MLE')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

x4, y4 = transition_util(test_c_lstm_gan_p_attr)

plt.subplot(224)
plt.bar(x4, y4)
plt.title(f'Melodies generated by C-LSTM-GAN')
plt.ylabel('Occurrences')
plt.xlabel('MIDI Number transition')
plt.ylim(0, 8000)

plt.tight_layout()
plt.savefig('./figures/transitions_without_kl.png')


# %% md

### Figure 11 & 12 Boxplots of distributions of drs, drn, and drns

# %%


def randomize_order(dat_attr, gen_attr, num_notes,
                    is_randomize_song_order=False,
                    is_randomize_note_order=False):
    distances = []
    for i in range(10000):
        if is_randomize_song_order:
            np.random.shuffle(dat_attr)

        if is_randomize_note_order:
            dat_attr = np.transpose(dat_attr)
            np.random.shuffle(dat_attr)
            dat_attr = np.transpose(dat_attr)

        dist = np.sum(np.abs(np.subtract(dat_attr, gen_attr)))
        distances.append(dist)

    return [dist / num_notes for dist in distances]


# %%

def randomize_util(dat_attr, gen_attr):
    num_notes = dat_attr.shape[0] * dat_attr.shape[1]
    dist_dat_gen = np.sum(np.abs(np.subtract(dat_attr, gen_attr)))

    dist_avg_random_songs = randomize_order(dat_attr,
                                            gen_attr,
                                            num_notes,
                                            is_randomize_song_order=True,
                                            is_randomize_note_order=False)
    dist_avg_random_songs.insert(0, dist_dat_gen / num_notes)

    dist_avg_random_notes_order = randomize_order(dat_attr,
                                                  gen_attr,
                                                  num_notes,
                                                  is_randomize_song_order=False,
                                                  is_randomize_note_order=True)
    dist_avg_random_notes_order.insert(0, dist_dat_gen / num_notes)

    dist_avg_random_songs_and_notes = randomize_order(dat_attr,
                                                      gen_attr,
                                                      num_notes,
                                                      is_randomize_song_order=True,
                                                      is_randomize_note_order=True)

    dist_avg_random_songs_and_notes.insert(0, dist_dat_gen / num_notes)

    return (dist_avg_random_songs,
            dist_avg_random_notes_order,
            dist_avg_random_songs_and_notes)


# %%

x2, y2, z2 = randomize_util(test_dat_d_attr, test_c_hybrid_gan_d_attr)

# %%

x2[0], y2[0], z2[0], np.mean(x2[1:]), np.mean(y2[1:]), np.mean(z2[1:])

# %%

x3, y3, z3 = randomize_util(test_dat_r_attr, test_c_hybrid_gan_r_attr)

# %%

x3[0], y3[0], z3[0], np.mean(x3[1:]), np.mean(y3[1:]), np.mean(z3[1:])

# %%


fig = plt.figure(num=None, figsize=(8, 4), dpi=400, facecolor='w', edgecolor='k')

plt.rc('axes', titlesize=14, labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize

plt.boxplot([x2, y2, z2])
correct = plt.plot([None, x2[0], y2[0], z2[0]], 'o', color='red', label='Correct Order')
plt.legend(handles=correct, loc=1)
my_xticks = ['Random song\norder',
             'Random note order\nwithin song',
             'Random song and\nnote order']
plt.xticks(np.array([1, 2, 3]), my_xticks)
plt.ylabel('Average note duration distance\nbetween generated sequences\n and sequences from the dataset')

plt.legend()
plt.tight_layout()
plt.savefig('./figures/duration_distance_boxplot.png')

# %%


fig = plt.figure(num=None, figsize=(8, 4), dpi=400, facecolor='w', edgecolor='k')

plt.rc('axes', titlesize=14, labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize

plt.boxplot([x3, y3, z3])
correct = plt.plot([None, x3[0], y3[0], z3[0]], 'o', color='red', label='Correct Order')
plt.legend(handles=correct, loc=1)
my_xticks = ['Random song\norder',
             'Random note order\nwithin song',
             'Random song and\nnote order']
plt.xticks(np.array([1, 2, 3]), my_xticks)
plt.ylabel('Average rest distance between\ngenerated sequences and\nsequences from the dataset')

plt.legend()
plt.tight_layout()
plt.savefig('./figures/rest_distance_boxplot.png')

# %%


# end of notebook
