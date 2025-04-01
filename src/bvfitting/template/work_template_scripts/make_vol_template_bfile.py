#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/03/26 09:51:11

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import meshio as io
import cheartio as chio


def get_surface_mesh(mesh):
    ien = mesh.cells[0].data

    if ien.shape[1] == 3:   # Assuming tetra
        array = np.array([[0,1],[1,2],[2,0]])
        nelems = np.repeat(np.arange(ien.shape[0]),3)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]


    elif ien.shape[1] == 4:   # Assuming tetra
        array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
        nelems = np.repeat(np.arange(ien.shape[0]),4)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]

    elif ien.shape[1] == 27:   # Assuming hex27
        array = np.array([[0,1,5,4,8,17,12,16,22],
                          [1,2,6,5,9,18,13,17,21],
                          [2,3,7,6,10,19,14,18,23],
                          [3,0,4,7,11,16,15,19,20],
                          [0,1,2,3,8,9,10,11,24],
                          [4,5,6,7,12,13,14,15,25]])
        nelems = np.repeat(np.arange(ien.shape[0]),6)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]
        
    return belem, bfaces


def create_submesh(mesh, map_mesh_submesh_elems):
    submesh_elems = mesh.cells[0].data[map_mesh_submesh_elems]
    submesh_xyz = np.zeros([len(np.unique(submesh_elems)),3])
    map_mesh_submesh = np.ones(mesh.points.shape[0], dtype=int)*-1
    map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
    child_elems_new = np.zeros(submesh_elems.shape, dtype=int)

    cont = 0
    for e in range(submesh_elems.shape[0]):
        for i in range(submesh_elems.shape[1]):
            if map_mesh_submesh[submesh_elems[e,i]] == -1:
                child_elems_new[e,i] = cont
                submesh_xyz[cont] = mesh.points[submesh_elems[e,i]]
                map_mesh_submesh[submesh_elems[e,i]] = cont
                map_submesh_mesh[cont] = submesh_elems[e,i]
                cont += 1
            else:
                child_elems_new[e,i] = map_mesh_submesh[submesh_elems[e,i]]

    submesh = io.Mesh(submesh_xyz, {mesh.cells[0].type: child_elems_new})
    return submesh, map_mesh_submesh, map_submesh_mesh


def create_submesh_bdata(submesh, mesh_bdata, map_mesh_submesh, map_submesh_mesh_elems, method):
    if method == 'parent':  # This means it will only look for the faces defined in the parent mesh
        belem = map_submesh_mesh_elems[mesh_bdata[:,0]]
        submesh_marker = belem >= 0
        belem = belem[submesh_marker]
        bfaces = map_mesh_submesh[mesh_bdata[submesh_marker,1:-1]]
        marker = mesh_bdata[submesh_marker,-1]

    elif method == 'boundary':  # This will look for the boundaries of the submesh and compare it with the parent to get markers
        belem, bfaces = get_surface_mesh(submesh)

        # Create face marker using bv mesh
        nb = np.unique(mesh_bdata[:,-1])

        marker = np.zeros(bfaces.shape[0], dtype=int)
        for b in nb:
            b_ien = map_mesh_submesh[mesh_bdata[mesh_bdata[:,-1]== b, 1:-1]]
            b_ien = b_ien[np.min(b_ien >= 0, axis=1)]
            marker[np.sum(np.isin(bfaces, b_ien), axis=1) == 3] = b

    bdata = np.hstack([belem[:,None], bfaces, marker[:,None]])

    return bdata



template = io.read('src/bvfitting/template/template.vtu')
vol_ien = np.load('src/bvfitting/template/volume_template_ien.npy').squeeze()

mesh = io.Mesh(template.points, {'tetra': vol_ien})
io.write('check.vtu', mesh)

labels = template.cell_data['Region'][0]
nlabels = np.max(labels) + 1

surf_ien = template.cells[0].data

#TODO need to find the inner face
belem, bfaces = get_surface_mesh(mesh)
new_labels = np.zeros(len(bfaces), dtype=int)
for i in range(nlabels):
    surf = surf_ien[labels == i]
    bnodes = np.unique(surf)
    marker = np.isin(bfaces, bnodes)
    count = np.sum(marker, axis=1)
    new_labels[count==3] = i

bdata = np.column_stack((belem, bfaces, new_labels))


# Create submesh
submesh, map_mesh_submesh, map_submesh_mesh = create_submesh(mesh, np.ones(mesh.cells[0].data.shape[0], dtype=bool))
submesh_bdata = create_submesh_bdata(submesh, bdata, map_mesh_submesh, map_submesh_mesh, 'boundary')

pv_elems = np.loadtxt('src/bvfitting/template/pv_elems.txt', dtype=int, skiprows=1, usecols=1, delimiter=',')
tv_elems = np.loadtxt('src/bvfitting/template/tv_elems.txt', dtype=int, skiprows=1, usecols=1, delimiter=',')
av_elems = np.loadtxt('src/bvfitting/template/av_elems.txt', dtype=int, skiprows=1, usecols=1, delimiter=',')
mv_elems = np.loadtxt('src/bvfitting/template/mv_elems.txt', dtype=int, skiprows=1, usecols=1, delimiter=',')

submesh_bdata[pv_elems, -1] = 9
submesh_bdata[tv_elems, -1] = 10
submesh_bdata[av_elems, -1] = 11
submesh_bdata[mv_elems, -1] = 12

chio.write_bfile('src/bvfitting/template/volume_template', submesh_bdata)
chio.write_mesh('src/bvfitting/template/volume_template', submesh.points, submesh.cells[0].data)

