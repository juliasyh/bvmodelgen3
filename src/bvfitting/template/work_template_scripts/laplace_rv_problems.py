#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/03/26 10:28:53

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
import meshio as io
import cheartio as chio
import dolfinxio as dxio
from uvcgen.LaplaceProblem import LaplaceProblem
from scipy.spatial import cKDTree

def sigmoid_func(x, a, b):
    return 1 / (1 + np.exp(-a*(x-b)))

mesh = chio.read_mesh('src/bvfitting/template/volume_template', meshio=True)
bdata = chio.read_bfile('src/bvfitting/template/volume_template')
apex_elems = [594, 595, 596, 597, 598, 599, 607, 5129]

dx_mesh, dx_mt = dxio.read_meshio_mesh(mesh, bdata, clean=False)
LapSolver = LaplaceProblem(dx_mesh, dx_mt)
corr, icorr = dxio.find_vtu_dx_mapping(dx_mesh)

# Sep to valves
bcs_marker = {'face': {1: 0.0, 8: 0.0,
                      9: 1.0, 10: 1.0}}

lap = LapSolver.solve(bcs_marker)
lap = lap.x.petsc_vec.array[corr]

mesh.point_data['lap'] = lap

rvlv_transition = sigmoid_func(lap, -30, 0.05)
rvlv_transition = (rvlv_transition - np.min(rvlv_transition)) / (np.max(rvlv_transition) - np.min(rvlv_transition))

mesh.point_data['lap'] = rvlv_transition
io.write('check.vtu', mesh)


# Valves to apex
bdata_apex = bdata.copy()
bdata_apex[apex_elems,-1] = 13

dx_mesh, dx_mt = dxio.read_meshio_mesh(mesh, bdata_apex, clean=False)
LapSolver = LaplaceProblem(dx_mesh, dx_mt)
corr, icorr = dxio.find_vtu_dx_mapping(dx_mesh)

bridge_elems = [468,582,2528,2555,2735,2742,2782,5032,5065,5130,
                268, 270, 4113, 3093, 3926]
point_bcs = {}
for i in bridge_elems:
    point_bcs[tuple(mesh.points[i])] = 1.0

bcs_marker = {'face': {9: 1.0, 10: 1.0,
                      11: 1.0, 12: 1.0,
                      13: 0.0}, 
                'point': point_bcs}

lap = LapSolver.solve(bcs_marker)
lap = lap.x.petsc_vec.array[corr]

val = 0.9
x = np.linspace(0, 1, 100)

valve_transition = sigmoid_func(lap, 60, val)
valve_transition = (valve_transition - np.min(valve_transition)) / (np.max(valve_transition) - np.min(valve_transition))

mesh.point_data['lap'] = valve_transition 
io.write('check.vtu', mesh)

template = io.read('src/bvfitting/template/template.vtu')
xyz = template.points
ien = template.cells[0].data

tree = cKDTree(xyz)
dist, idx = tree.query(mesh.points)

rv_distance_field = np.zeros(len(xyz))
rv_distance_field[idx] = valve_transition + rvlv_transition
rv_distance_field[rv_distance_field > 1] = 1
rv_distance_field[rv_distance_field < 0] = 0

template.point_data['rv_distance_field'] = rv_distance_field
io.write('src/bvfitting/template/template.vtu', template)
io.write('check.vtu', template)