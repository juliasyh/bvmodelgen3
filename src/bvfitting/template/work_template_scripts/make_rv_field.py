#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/03/13 10:20:44

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
import meshio as io

mesh = io.read('src/bvfitting/template/template.vtu')
valve_elems = np.loadtxt('src/bvfitting/template/valve_elems_mod.txt', dtype = int)

xyz = mesh.points
ien = mesh.cells[0].data
surfs = mesh.cell_data['Region'][0]

labels = {'av': 5, 
          'mv': 4, 
          'pv': 7, 
          'tv': 6, 
          'lv_epi': 8, 
          'rv_epi': 3,
          'lv_endo': 0,
          'rv_septum': 1,
          'rv_endo': 2,
          'lv_rv': 9
          }

# Grab rv boundary nodes
pv_nodes = np.unique(ien[surfs == labels['pv']])
tv_nodes = np.unique(ien[surfs == labels['tv']])
rv_epi_nodes = np.unique(ien[surfs == labels['rv_epi']])
rv_septum_nodes = np.unique(ien[surfs == labels['rv_septum']])
rv_endo_nodes = np.unique(ien[surfs == labels['rv_endo']])
lv_rv_nodes = np.unique(ien[surfs == labels['lv_rv']])
lv_endo_nodes = np.unique(ien[surfs == labels['lv_endo']])
lv_epi_nodes = np.unique(ien[surfs == labels['lv_epi']])
valve_nodes = np.unique(ien[valve_elems])

rv_pv_nodes = np.intersect1d(rv_epi_nodes, pv_nodes)
rv_tv_nodes = np.intersect1d(rv_epi_nodes, tv_nodes)
rv_epi_lvrv_nodes = np.intersect1d(rv_epi_nodes, lv_rv_nodes)
rv_endo_lvrv_nodes = np.intersect1d(rv_endo_nodes, lv_rv_nodes)
rv_epi_valves_nodes = np.intersect1d(rv_epi_nodes, valve_nodes)

bnodes = np.concatenate([rv_pv_nodes, rv_tv_nodes, rv_epi_lvrv_nodes, rv_endo_lvrv_nodes, rv_epi_valves_nodes])

# calcualte distance
# Distance field from rv_lv
propagation = 10
distance_field = np.zeros(len(xyz))
distance_field[bnodes] = 1

nx = np.zeros(len(xyz))
for i in range(len(ien)):
    nx[ien[i]] += 3

# Propagation
for i in range(propagation):
    aux = distance_field.copy()
    for j in range(len(ien)):
        ax = np.sum(distance_field[ien[j]])
        aux[ien[j]] += ax
    aux[nx != 0] = aux[nx != 0] / nx[nx != 0]
    aux[bnodes] = 1
    distance_field = aux

# Cutoff
distance_field[distance_field > 1] = 1
mesh.point_data['rv_distance_field'] = distance_field
io.write('src/bvfitting/template/template.vtu', mesh)

# # Distance field from rv_lv
# bnodes = np.concatenate([rv_epi_lvrv_nodes, rv_endo_lvrv_nodes])
# propagation = 10
# distance_field_rvlv = np.zeros(len(xyz))
# distance_field_rvlv[bnodes] = 1

# nx = np.zeros(len(xyz))
# for i in range(len(ien)):
#     nx[ien[i]] += 3

# # Propagation
# for i in range(propagation):
#     aux = distance_field_rvlv.copy()
#     for j in range(len(ien)):
#         ax = np.sum(distance_field_rvlv[ien[j]])
#         aux[ien[j]] += ax
#     aux[nx != 0] = aux[nx != 0] / nx[nx != 0]
#     aux[bnodes] = 1
#     distance_field_rvlv = aux

# # Cutoff
# distance_field_rvlv[distance_field_rvlv > 1] = 1

# # Distance field from rv_valves
# bnodes = np.concatenate([rv_pv_nodes, rv_tv_nodes, rv_epi_valves_nodes])
# propagation = 20
# distance_field_rvvalves = np.zeros(len(xyz))
# distance_field_rvvalves[bnodes] = 1

# # Propagation
# for i in range(propagation):
#     aux = distance_field_rvvalves.copy()
#     for j in range(len(ien)):
#         ax = np.sum(distance_field_rvvalves[ien[j]])
#         aux[ien[j]] += ax
#     aux[nx != 0] = aux[nx != 0] / nx[nx != 0]
#     aux[bnodes] = 1
#     distance_field_rvvalves = aux

# distance_field_rvvalves[distance_field_rvvalves > 1] = 1

# distance_field = distance_field_rvlv + distance_field_rvvalves
# distance_field[distance_field > 1] = 1
# distance_field[distance_field < 0] = 0


# mesh.point_data['rv_distance_field'] = distance_field
# io.write('src/bvfitting/template/template.vtu', mesh)