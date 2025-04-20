import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# import pytorch3d
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
OrthographicCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)


def render_mesh_orthogonal(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().cuda())
    mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]),
                       2)  # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = OrthographicCameras(focal_length=cam_param['focal'],
                                  # principal_point=cam_param['princpt'],
                                  device='cuda',
                                  in_ndc=False,
                                  image_size=torch.LongTensor(render_shape).cuda().view(1, 2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0,
                                            faces_per_pixel=1)  # , perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
        device='cuda',
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=color,
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps


def render_mesh_perspective(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().cuda())
    mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]),
                       2)  # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                 principal_point=cam_param['princpt'],
                                 device='cuda',
                                 in_ndc=False,
                                 image_size=torch.LongTensor(render_shape).cuda().view(1, 2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0,
                                            faces_per_pixel=1)  # , perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
        device='cuda',
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=color,
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps