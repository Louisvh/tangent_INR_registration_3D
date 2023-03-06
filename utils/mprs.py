import numpy as np
import torch

from utils.general import *


def rotation_matrix(phi, axis):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(phi / 2)
    b,c,d = -axis * np.sin(phi / 2)
    aa, bb, cc, dd, ab, ac, ad, bc, bd, cd = a*a, b*b, c*c, d*d, a*b, a*c, a*d, b*c, b*d, c*d
    rotmat = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]) 
    return rotmat


def compute_parallel_transport(centerline, init_rot=0):
    tangents = np.gradient(centerline, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    V = np.zeros_like(centerline)
    V[0] = (tangents[0][1], -tangents[0][0], 0)
    V[0] = V[0] / np.linalg.norm(V[0])
    
    R = rotation_matrix(init_rot*np.pi/180, tangents[0])
    V[0] = np.dot(R, V[0])

    for i in range(centerline.shape[0] - 1):
        b = np.cross(tangents[i], tangents[i + 1])
        if np.linalg.norm(b) < 0.0001:
            V[i + 1] = V[i]
        else:
            b = b / np.linalg.norm(b)
            phi = np.arccos(np.dot(tangents[i], tangents[i + 1]))
            R = rotation_matrix(phi, b)
            V[i + 1] = np.dot(R, V[i])
   
    U = np.cross(tangents, V)

    return U, V

def draw_plane_from_normal_vecs(image, x, y, z, imagespacing, patchsize, patchspacing, u, v, interpolation='nearest'):
    """
    sample along two normal vectors u,v, rather than rotating a grid
    """
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    
    xs = (x + ((unra[0] - patchmargin[0]) * u[0] + (unra[1] - patchmargin[1]) * v[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + ((unra[0] - patchmargin[0]) * u[1] + (unra[1] - patchmargin[1]) * v[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + ((unra[0] - patchmargin[0]) * u[2] + (unra[1] - patchmargin[1]) * v[2]) * patchspacing[2]) / imagespacing[2]

    t_xs = torch.as_tensor(xs).reshape(tuple(patchsize)).float() / image.shape[0] *2 - 1
    t_ys = torch.as_tensor(ys).reshape(tuple(patchsize)).float() / image.shape[1] *2 - 1
    t_zs = torch.as_tensor(zs).reshape(tuple(patchsize)).float() / image.shape[2] *2 - 1
    t_grid = torch.stack([t_zs, t_ys, t_xs], axis=-1)

    if interpolation == 'linear':
        patch = torch.nn.functional.grid_sample(image[None,None,:], t_grid[None,:], align_corners=True)
    else:
        patch = torch.nn.functional.grid_sample(image[None, None, :], t_grid[None, :], align_corners=True, mode='nearest')

    reshaped_patch = patch.view(patchsize.tolist())
    return reshaped_patch


def get_imcoords_from_mmnormal_vecs(centers, uv_ind, imagespacing, patchsize, u, v):
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    xs = (x + uv_ind[:, 0] * u[:, 0] + uv_ind[:, 1] * v[:, 0]) / imagespacing[0]
    ys = (y + uv_ind[:, 0] * u[:, 1] + uv_ind[:, 1] * v[:, 1]) / imagespacing[1]
    zs = (z + uv_ind[:, 0] * u[:, 2] + uv_ind[:, 1] * v[:, 2]) / imagespacing[2]

    t_xs = xs.reshape((patchsize)).float()
    t_ys = ys.reshape((patchsize)).float()
    t_zs = zs.reshape((patchsize)).float()
    t_grid = torch.stack([t_xs, t_ys, t_zs], axis=-1)
    return t_grid


def vox_coords_to_mm(coords, spacing):
    """ Convert voxel coordinates to mm distance from the origin"""
    if np.size(coords) == 3:
        return [coords[i] * spacing[i] for i in range(3)]

    coords_mm = []
    for coord in coords:
        coords_mm.append([coord[i] * spacing[i] for i in range(3)])

    return np.array(coords_mm)


def mm_coords_to_vox(coords, spacing):
    """ Convert mm distance from the origin to voxel coordinates"""
    if np.size(coords) == 3:
        return [coords[i] / spacing[i] for i in range(3)]

    coords_mm = []
    for coord in coords:
        coords_mm.append([coord[i] / spacing[i] for i in range(3)])

    return np.array(coords_mm)


def mpr_generator(trace, datavol, orig_spacing, mpr_spacing, diameter, mask_planes=False, interp='linear', init_rot=0, maskval=0):
    """
    generate mpr from trace and data
    
    diameter: diameter of the MPR in voxels
    mask_planes: tubular mask to ensure rotational symmetry (size matches diameter)
    interp: interpolation order (nearest, [linear])
    """
    trace_mm = vox_coords_to_mm(trace, orig_spacing)
    U, V = compute_parallel_transport(trace_mm, init_rot)
    
    vox_dims = np.asarray([diameter, diameter, 1])
    
    mpr_container = np.zeros((len(trace_mm), diameter, diameter))
    
    Y, X = np.ogrid[:diameter, :diameter]
    center = ((diameter-1)/2, (diameter-1)/2)
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= diameter/2
    addmask = (dist_from_center > diameter/2)*maskval
    
    for v_step in range(len(trace_mm)):
        v_x, v_y, v_z = trace_mm[v_step]
        u=U[v_step]
        v=V[v_step]
        #print(u, v)

        plane=draw_plane_from_normal_vecs(datavol,v_x,v_y,v_z, orig_spacing, vox_dims, mpr_spacing, u, v, interpolation=interp)[:,:,0]
        if mask_planes:
            mpr_container[v_step,:] = plane * mask + addmask
        else:
            mpr_container[v_step,:] = plane
    
    return mpr_container


def smooth_gttrace(gttrace, fit_num=8, plotres=False, upsamplefac=1.0):
    trace_t = np.arange(int(gttrace.shape[0]))
    fit_x = np.polyfit(trace_t, gttrace[:,0],fit_num)
    fit_y = np.polyfit(trace_t, gttrace[:,1], fit_num)
    fit_z = np.polyfit(trace_t, gttrace[:,2], fit_num)
    
    poly_x = np.poly1d(fit_x)
    poly_y = np.poly1d(fit_y)
    poly_z = np.poly1d(fit_z)
    usamp_t = np.linspace(0,int(gttrace.shape[0]-1), int(gttrace.shape[0]*upsamplefac))
    gttrace_smoothed = np.asarray([poly_x(usamp_t), poly_y(usamp_t), poly_z(usamp_t)]).T
    if plotres:
        plt.plot(gttrace[:,0], gttrace[:,1])
        plt.plot(poly_x(usamp_t), poly_y(usamp_t))
        plt.plot(gttrace_smoothed[:,0], gttrace_smoothed[:,1])
        plt.show()
        plt.plot(gttrace[:,0], gttrace[:,2])
        plt.plot(poly_x(usamp_t), poly_z(usamp_t))
        plt.show()
    return gttrace_smoothed

    
def orig_voxtrace_to_new_spacing(trace, orig_spacing, new_spacing, smooth=True):
    trace_mm = vox_coords_to_mm(trace, orig_spacing)
    trace_new_spacing = mm_coords_to_vox(trace_mm, new_spacing)
    if smooth:
        trace_new_spacing = smooth_gttrace(trace_new_spacing)
    return trace_new_spacing    

def create_mpr(image, trace, diameter, orig_spacing=(1,1,1), mpr_spacing=(1,1,1), mask_planes=True, interp='linear', init_rot=0):

    trace_mm = vox_coords_to_mm(trace, orig_spacing)
    U, V = compute_parallel_transport(trace_mm, init_rot)
    
    vox_dims = np.asarray([diameter, diameter, 1])
    
    mpr_container = np.zeros((trace_mm.shape[0], diameter, diameter))
    
    Y, X = np.ogrid[:diameter, :diameter]
    center = (int(diameter/2), int(diameter/2))
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= diameter/2
    
    for v_step in range(trace_mm.shape[0]):
        v_x, v_y, v_z = trace_mm[v_step,:]
        u=U[v_step]
        v=V[v_step]

        plane=draw_plane_from_normal_vecs(image,v_x,v_y,v_z, orig_spacing, vox_dims, mpr_spacing, u, v, interpolation=interp)[:,:,0]
        if mask_planes:
            mpr_container[v_step,:] = plane * mask
        else:
            mpr_container[v_step,:] = plane
    
    return mpr_container
 

