#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt

#%%

nb_argv = 3
if nb_argv > 2:
    fig_xy, axes_xy = plt.subplots(1, nb_argv-1, sharex=True, sharey=True)
    for i in range(1, nb_argv):
        sns.kdeplot(east)