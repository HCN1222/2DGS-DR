#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math, time
import torch.nn.functional as F
import diff_surfel_rasterization_c3
import diff_surfel_rasterization_c4
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import numpy as np
from utils.point_utils import depth_to_normal


def decode_allmap(allmap,pipe,viewpoint_camera,rets):
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

# rayd: x,3, from camera to world points
# normal: x,3
# all normalized
def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl

def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)

def render_env_map(pc: GaussianModel):
    env_cood1 = sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap)
    env_cood2 = sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap)
    return {'env_cood1': env_cood1, 'env_cood2': env_cood2}

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, initial_stage = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    def get_setting_surfel(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings

    
    # init rasterizer with various channels
    Setting_c3 = diff_surfel_rasterization_c3.GaussianRasterizationSettings
    Setting_c4 = diff_surfel_rasterization_c4.GaussianRasterizationSettings
    
    rasterizer_c3 = diff_surfel_rasterization_c3.GaussianRasterizer(get_setting_surfel(Setting_c3))
    rasterizer_c4 = diff_surfel_rasterization_c4.GaussianRasterizer(get_setting_surfel(Setting_c4))

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii, allmap = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            # bg_map = bg_map_const
            )

        rets={
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}
        
        decode_allmap(allmap,pipe,viewpoint_camera,rets)
        return rets

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    refl_ratio = pc.get_refl
    # print(f'shape of normals: {normals.shape}')
    # print(f'shape of refl_ratio: {refl_ratio.shape}')
    input_ts = torch.cat([torch.zeros(scales.size(0),3, device='cuda'), refl_ratio], dim=-1) # x,4

    # bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii, allmap = rasterizer_c4(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        # bg_map = bg_map)
    )
    # MODIFY INDEXES
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[3:4,...] #
    rets={
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}
        
    decode_allmap(allmap,pipe,viewpoint_camera,rets)
    normal_map = rets['rend_normal'].permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    # print( f'viewpoint_camera.HWK: {viewpoint_camera.HWK}' )
    # print( f'viewpoint_camera.R: {viewpoint_camera.R}' )
    # print( f'viewpoint_camera.T: {viewpoint_camera.T}' )
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    
    # print(refl_strength.max())
    # # print(normal_map.requires_grad)
    # print(refl_color.requires_grad)
    # print(refl_color)
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii
    }
    rets.update(results)
        
    return rets
