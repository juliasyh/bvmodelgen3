#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/01 15:07:06

@author: Javiera Jilberto Vallejos 
'''

import meshio as io
import numpy as np

template = io.read('../template_og.vtu')

pv_elems = np.loadtxt('pv_template_elems.csv', dtype=int, skiprows=1, usecols=1, delimiter=',')
tv_elems = np.loadtxt('tv_template_elems.csv', dtype=int, skiprows=1, usecols=1, delimiter=',')
av_elems = np.loadtxt('av_template_elems.csv', dtype=int, skiprows=1, usecols=1, delimiter=',')
mv_elems = np.loadtxt('mv_template_elems.csv', dtype=int, skiprows=1, usecols=1, delimiter=',')

labels = template.cell_data['Region'][0]
labels[av_elems] = 10
labels[pv_elems] = 11
labels[mv_elems] = 12
labels[tv_elems] = 13

template.cell_data['Region'] = [labels]
io.write('../template.vtu', template, binary=True)
np.savetxt('../surface_region_mod.txt', labels.reshape((-1,1)), fmt='%d')