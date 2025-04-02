#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/02 09:12:58

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import meshio as io
import pyvista as pv

mesh = pv.Plane()
print(mesh)

mesh.point_data.clear()
mesh.plot(show_edges=True)