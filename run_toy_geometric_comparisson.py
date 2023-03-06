from utils import general, toy_examples, mprs
from objectives import ssim
from models import boundingbox_model as models
from models import single_mpr_model as mpr_models
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import torch

####################################
#
# To use this code with real data, load your own img_t0, img_t1 and supply
# a relevant curve as S_func (i.e. the center of the tangent coordinate system).
# 
# We initialize a toy example of blobs surrounding a polynomial curve.
# Any supplied mask is only used for visualization: the mask used during optimization
# is derived from the supplied centerline function and mask radius _maskr_.
#
# The majority of this script deals with plotting and comparing results. 
# To optimize a single tangent-space registration INR, simply use:
#
#    mpr_ImpReg = mpr_models.MPRImplicitRegistrator(S_func, img_t1, img_t0, **kwargs)
#    mpr_ImpReg.fit()
#    mpr_ImpReg.savenets()
#
####################################

kwargs = {}
kwargs["verbose"] = False
kwargs["jacobian_regularization"] = True
kwargs["network_type"] = "SIREN"  
kwargs["layers"] = [3, 256, 256, 256, 3]
kwargs["omega"] = 64
kwargs["optimizer"] = "Adam"
kwargs["lr"] = 1e-4
kwargs["batch_size"] = 10000
kwargs["experiment_key"] = 'toy_debug_geo'
kwargs["save_folder"] = 'results/toy_debug/'

kwargs["epochs"] = 2501
kwargs["alpha_jacobian"] = 5e-2
eval_every_nth = 25
seedstart = 42
nseeds = 8  # 32  # increase for more accurate IQR
vminmaxbound = 1
ssim_window = 5

# construct toy example
print('constructing toy example...')
zdim = 75
zloc = zdim // 2
ydim = 255
xdim = 256
maskr = 25
mpr_diam = 2*maskr
m_widths = (10,20)
m_subfac = 0.5
n_steps = 512
time_delay = 3.14/4
(
    img_t1, 
    img_t0, 
    voxel_size,
    mask
) = toy_examples.load_simple_toy_example('S', widths=m_widths,shape=(zdim,ydim,xdim), maskwidth=maskr, subfac=m_subfac, max_t = 1.9, b_delay=time_delay)

t,x,y,_,_,dnorm = toy_examples.get_toy_func('S', (zdim,ydim,xdim), n_steps, max_t=1.7)
z = np.ones_like(x)*zdim/2
arclen = np.sum(dnorm)
mpr_plotratio = arclen / zdim
S_func = np.stack([z,y,x]).T

print(f'S_func shape: {S_func.shape}')
print(f'img_t1.shape: {img_t1.shape}')

f = plt.figure(figsize=(20,8), layout='constrained')
spec = f.add_gridspec(3,6)

imspace_axes = []
reformed_imspace_axes = []
tangentspace_axes = []
for i,ax in enumerate([imspace_axes, reformed_imspace_axes, tangentspace_axes]):
    ax.append(f.add_subplot(spec[i,0]))
    ax.append(f.add_subplot(spec[i,1]))
    ax.append(f.add_subplot(spec[i,2]))
    ax.append(f.add_subplot(spec[i,3]))
plot_ax = f.add_subplot(spec[:,4])
plot_ax2 = f.add_subplot(spec[:,5])
a = [imspace_axes, tangentspace_axes, reformed_imspace_axes, [plot_ax], [plot_ax2]]

a[0][0].imshow((img_t1[zdim//2,:,:]+(1-mask[zdim//2,:,:])).T, cmap='gray')
a[0][1].imshow((img_t0[zdim//2,:,:]+(1-mask[zdim//2,:,:])).T, cmap='gray')
a[1][0].imshow((img_t1[zdim//2,:,:]+(1-mask[zdim//2,:,:])).T, cmap='gray')
a[1][1].imshow((img_t0[zdim//2,:,:]+(1-mask[zdim//2,:,:])).T, cmap='gray')
a[0][0].plot(y,x)
a[0][1].plot(y,x)

mpr_t0 = mprs.mpr_generator(S_func, img_t0, voxel_size, (1,1,1), mpr_diam*2)
mpr_t1 = mprs.mpr_generator(S_func, img_t1, voxel_size, (1,1,1), mpr_diam*2)
t_mask = torch.as_tensor(mask)*1.0
mpr_sammask = mprs.mpr_generator(S_func, t_mask, voxel_size, (1,1,1), mpr_diam*2)
catmpr_t0 = mpr_t0[:, :, mpr_t0.shape[2] // 2].T
catmpr_t1 = mpr_t1[:, :, mpr_t0.shape[2] // 2].T
catmpr_sammask = mpr_sammask[:, :, mpr_sammask.shape[2] // 2].T
a[1][1].imshow(catmpr_t0, cmap='gray', aspect=mpr_plotratio)
a[1][0].imshow(catmpr_t1, cmap='gray', aspect=mpr_plotratio)
a[2][1].imshow(catmpr_t0, cmap='gray', aspect=mpr_plotratio)
a[2][0].imshow(catmpr_t1, cmap='gray', aspect=mpr_plotratio)
a[1][2].imshow(catmpr_sammask, cmap='gray', aspect=mpr_plotratio)

all_jacdets_S = []
all_maes_trad = []
all_ssim_trad = []
all_resviz_trad = []
all_diffviz_trad = []
all_reforesviz_trad = []
all_refodiffviz_trad = []
all_maes_mpr = []
all_ssim_mpr = []
all_resviz_mpr = []
all_diffviz_mpr = []
mask = torch.as_tensor(mask)
kwargs["mask"] = mask
ImpReg = models.ImplicitRegistrator(img_t1, img_t0, **kwargs)
for mseed in range(seedstart,seedstart+nseeds):
    print('\n-----\n')
    print(f'seed: {mseed} ({mseed+1-seedstart}/{nseeds})\n')
    kwargs["seed"] = mseed

    ImpReg = models.ImplicitRegistrator(img_t1, img_t0, **kwargs)
    kwargs["manifold_diam"] = mpr_diam
    kwargs["voxel_spacing"] = tuple(voxel_size)
    mpr_ImpReg = mpr_models.MPRImplicitRegistrator(S_func, img_t1, img_t0, **kwargs)
    evalcoords_imspace = mpr_ImpReg.mpr_coords_to_imspace(mpr_ImpReg.possible_coordinate_tensor)
    evalcoords_normspace = mpr_ImpReg.imspace_to_normspace(evalcoords_imspace)

    harmonize_mask_coords = True
    maskcoords = ImpReg.possible_coordinate_tensor
    if harmonize_mask_coords:
        ImpReg.possible_coordinate_tensor = evalcoords_normspace
        evalmask = mpr_ImpReg.mpr_mask
    else:
        evalmask = ImpReg.mask

    #optimize in image coords
    trad_mae_results = []
    trad_ssim_results = []
    torch.manual_seed(mseed)

    # prep imspace visualization coords
    nx, ny = (ydim, xdim)
    x_ind = torch.linspace(-1, 1, nx)
    y_ind = torch.linspace(-1, 1, ny)
    c_x, c_y = torch.meshgrid([x_ind, y_ind])
    c_z = torch.ones_like(c_x) * (zloc / img_t0.shape[0] * 2 - 1)
    mcoords = torch.stack([c_z, c_x, c_y], axis=-1).cuda()

    mprspace_maskcoords = evalcoords_normspace
    mprspace_fixed_canvas = evalmask.cpu().float().numpy().copy()
    cshape = mprspace_fixed_canvas.shape
    mae_corfac = cshape[0]*cshape[1]*cshape[2] / np.sum(mprspace_fixed_canvas)
    fixed_samples = ImpReg(mprspace_maskcoords, (mprspace_maskcoords.shape[0],1), ret_fixed=True)
    mprspace_fixed_canvas[np.where(mprspace_fixed_canvas > 0)] = fixed_samples[:, 0] 
    mprspace_fixed_tensor = torch.Tensor(mprspace_fixed_canvas).cuda()
    # split to reduce GPU memory footprint
    mprspace_maskcoords1 = mprspace_maskcoords[:mprspace_maskcoords.shape[0]//2,:]
    mprspace_maskcoords2 = mprspace_maskcoords[mprspace_maskcoords.shape[0]//2:,:]
    for i in tqdm.tqdm(range(ImpReg.epochs)):
        if i % eval_every_nth == 0:
            with torch.no_grad():
                trad_sam1 = ImpReg(mprspace_maskcoords1, (mprspace_maskcoords1.shape[0],1))
                trad_sam2 = ImpReg(mprspace_maskcoords2, (mprspace_maskcoords2.shape[0],1))
                trad_sam = np.concatenate([trad_sam1, trad_sam2], axis=0)
                trad_sam_canvas = evalmask.cpu().float().numpy().copy()
                trad_sam_canvas[np.where(trad_sam_canvas > 0)] = trad_sam[:, 0] 
                trad_sam_tensor = torch.Tensor(trad_sam_canvas).cuda()
                trad_diffim = trad_sam_tensor - mprspace_fixed_tensor
                trad_mae = torch.mean(torch.sqrt(trad_diffim*trad_diffim))
                trad_mae_results.append(trad_mae.cpu().numpy()*mae_corfac)

                trad_ssim = ssim.ssim3D(
                        trad_sam_tensor[None, None, :, maskr:-maskr, maskr:-maskr],
                        mprspace_fixed_tensor[None, None, :, maskr:-maskr, maskr:-maskr],
                        window_size=ssim_window)
                trad_ssim_results.append(trad_ssim.cpu().numpy())

        ImpReg.training_iteration(i)
    ImpReg.savenets()
    all_maes_trad.append(trad_mae_results)
    all_ssim_trad.append(trad_ssim_results)

    with torch.no_grad():
        sam_im = ImpReg(mcoords.reshape((nx * ny, 3)), output_shape=(nx, ny))
        sam_im_fixed = ImpReg(mcoords.reshape((nx * ny, 3)), output_shape=(nx, ny), ret_fixed=True)
    del(ImpReg)

    #paint harmonized mask back into original imspace (just for plotting)
    imspace_canvas = torch.zeros_like(mpr_ImpReg.moving_image)
    imspace_mprspace_maskcoords = evalcoords_imspace
    for i in range(3):
        imspace_mprspace_maskcoords[:, i] = torch.clip(imspace_mprspace_maskcoords[:, i], 0,
                                                    imspace_canvas.shape[i] - 1)
    imspace_mprspace_maskcoords = torch.round(imspace_mprspace_maskcoords).long()
    imspace_canvas[imspace_mprspace_maskcoords[:, 0],
                    imspace_mprspace_maskcoords[:, 1],
                    imspace_mprspace_maskcoords[:, 2]] = 1

    m_imspace_canvas = imspace_canvas.cpu().numpy()[zloc, :, :]
    slicemask = (((m_imspace_canvas > 0) * 0.5) + 0.5)

    all_resviz_trad.append((sam_im * slicemask).T)
    all_diffviz_trad.append(((sam_im-sam_im_fixed) * (slicemask>0.75)).T)

    refo_mask_canvas = trad_sam_tensor.cpu().numpy()
    refo_fixed_canvas = mprspace_fixed_tensor.cpu().numpy()

    mdiam = mpr_diam
    reformed_sam_trad = refo_mask_canvas[:, refo_mask_canvas.shape[2] // 2, :].T
    reformed_fixed_trad = refo_fixed_canvas[:, refo_fixed_canvas.shape[2] // 2, :].T
    all_reforesviz_trad.append(reformed_sam_trad)
    all_refodiffviz_trad.append(reformed_sam_trad-reformed_fixed_trad)

    for i in [0]:
        a[i][1].imshow(img_t0[zloc, :, :].T.cpu().numpy()*slicemask.T, cmap='gray', vmin=0, vmax=vminmaxbound)
        a[i][0].imshow(img_t1[zloc, :, :].T.cpu().numpy()*slicemask.T, cmap='gray', vmin=0, vmax=vminmaxbound)

    # mpr_space version
    mpr_mae_results = []
    mpr_ssim_results = []
    torch.manual_seed(mseed)
    mpr_maskcoords = mpr_ImpReg.possible_coordinate_tensor
    # split to reduce GPU memory footprint
    mpr_maskcoords1 = mpr_maskcoords[:mpr_maskcoords.shape[0]//2,:]
    mpr_maskcoords2 = mpr_maskcoords[mpr_maskcoords.shape[0]//2:,:]

    for i in tqdm.tqdm(range(mpr_ImpReg.epochs)):
        if i % eval_every_nth == 0:
            with torch.no_grad():
                mpr_sam1 = mpr_ImpReg(mpr_maskcoords1, (mpr_maskcoords1.shape[0],1))
                mpr_sam2 = mpr_ImpReg(mpr_maskcoords2, (mpr_maskcoords2.shape[0],1))
                mpr_sam = np.concatenate([mpr_sam1, mpr_sam2], axis=0)
                mpr_sam_canvas = evalmask.cpu().float().numpy().copy()
                mpr_sam_canvas[np.where(mpr_sam_canvas > 0)] = mpr_sam[:, 0] 
                mpr_sam_tensor = torch.Tensor(mpr_sam_canvas).cuda()
                mpr_diffim = mpr_sam_tensor - mprspace_fixed_tensor
                mpr_mae = torch.mean(torch.sqrt(mpr_diffim*mpr_diffim))
                mpr_mae_results.append(mpr_mae.cpu().numpy()*mae_corfac)

                mpr_ssim = ssim.ssim3D(
                        mpr_sam_tensor[None, None, :, maskr:-maskr, maskr:-maskr],
                        mprspace_fixed_tensor[None, None, :, maskr:-maskr, maskr:-maskr],
                        window_size=ssim_window)
                mpr_ssim_results.append(mpr_ssim.cpu().numpy())
        mpr_ImpReg.training_iteration(i)
    mpr_ImpReg.savenets()
    all_maes_mpr.append(mpr_mae_results)
    all_ssim_mpr.append(mpr_ssim_results)

    mpr_mask_canvas = mpr_sam_tensor.cpu().numpy()
    mpr_fixed_canvas = mprspace_fixed_tensor.cpu().numpy()

    mdiam = mpr_diam
    double_mpr = mpr_mask_canvas[:, mpr_mask_canvas.shape[2] // 2, :].T
    double_mpr_fixed = mpr_fixed_canvas[:, mpr_fixed_canvas.shape[2] // 2, :].T

    all_resviz_mpr.append(double_mpr)
    all_diffviz_mpr.append(double_mpr-double_mpr_fixed)

a[0][0].set_title('Source')
a[0][1].set_title('Target')
a[0][2].set_title('Result')
a[0][3].set_title('Error')

a[0][0].set_ylabel('Image space INR')
a[2][0].set_ylabel('Reformed image space')
a[1][0].set_ylabel('Tangent space INR')
n_iqr = int(nseeds//4)
allmae_trad = np.stack(all_maes_trad, axis=0)
allmae_trad_sorted = np.sort(allmae_trad, axis=0)
trad_mae_sortind = np.argsort(allmae_trad[:,-1])
min_trad_mae_results = np.min(allmae_trad, axis=0)
print(f'min_trad_mae_results[-1]: {min_trad_mae_results[-1]}')
max_trad_mae_results = np.max(allmae_trad, axis=0)
print(f'max_trad_mae_results[-1]: {max_trad_mae_results[-1]}')
print(f'type max_trad_mae_results[-1]: {type(max_trad_mae_results[-1])}')
median_trad_mae_results = np.median(allmae_trad, axis=0)
allmae_mpr = np.stack(all_maes_mpr, axis=0)
allmae_mpr_sorted = np.sort(allmae_mpr, axis=0)
mpr_mae_sortind = np.argsort(allmae_mpr[:,-1])
min_mpr_mae_results = np.min(allmae_mpr, axis=0)
max_mpr_mae_results = np.max(allmae_mpr, axis=0)
median_mpr_mae_results = np.median(allmae_mpr, axis=0)
xlin = np.linspace(0,kwargs["epochs"]-1,len(trad_mae_results))
a[3][0].plot(xlin, median_trad_mae_results, c='xkcd:blue', label='Image space INR')
a[3][0].fill_between(xlin, min_trad_mae_results, max_trad_mae_results,
        alpha=0.1, color='xkcd:blue')
a[3][0].fill_between(xlin, allmae_trad_sorted[n_iqr,:], allmae_trad_sorted[-1-n_iqr,:],
        alpha=0.3, color='xkcd:blue', label='IQR')
a[3][0].plot(xlin, median_mpr_mae_results, c='xkcd:orange', label='Tangent space INR')
a[3][0].fill_between(xlin, min_mpr_mae_results, max_mpr_mae_results,
        alpha=0.1, color='xkcd:orange')
a[3][0].fill_between(xlin, allmae_mpr_sorted[n_iqr,:], allmae_mpr_sorted[-1-n_iqr,:],
        alpha=0.3, color='xkcd:orange', label='IQR')

allssim_trad = 1-np.stack(all_ssim_trad, axis=0)
allssim_trad_sorted = np.sort(allssim_trad, axis=0)
trad_ssim_sortind = np.argsort(allssim_trad[:,-1])
ssimmedind_trad = trad_ssim_sortind[trad_ssim_sortind.shape[0]//2]
min_trad_ssim_results = np.min(allssim_trad, axis=0)
max_trad_ssim_results = np.max(allssim_trad, axis=0)
median_trad_ssim_results = np.median(allssim_trad, axis=0)
allssim_mpr = 1-np.stack(all_ssim_mpr, axis=0)
allssim_mpr_sorted = np.sort(allssim_mpr, axis=0)
mpr_ssim_sortind = np.argsort(allssim_mpr[:,-1])
ssimmedind_mpr = mpr_ssim_sortind[mpr_ssim_sortind.shape[0]//2]
min_mpr_ssim_results = np.min(allssim_mpr, axis=0)
max_mpr_ssim_results = np.max(allssim_mpr, axis=0)
median_mpr_ssim_results = np.median(allssim_mpr, axis=0)

a[4][0].plot(xlin, median_trad_ssim_results, c='xkcd:blue', label='Image space INR')
a[4][0].fill_between(xlin, min_trad_ssim_results, max_trad_ssim_results,
        alpha=0.1, color='xkcd:blue')
a[4][0].fill_between(xlin, allssim_trad_sorted[n_iqr,:], allssim_trad_sorted[-1-n_iqr,:],
        alpha=0.3, color='xkcd:blue', label=f'IQR')
a[4][0].plot(xlin, median_mpr_ssim_results, c='xkcd:orange', label='Tangent space INR')
a[4][0].fill_between(xlin, min_mpr_ssim_results, max_mpr_ssim_results,
        alpha=0.1, color='xkcd:orange')
a[4][0].fill_between(xlin, allssim_mpr_sorted[n_iqr,:], allssim_mpr_sorted[-1-n_iqr,:],
        alpha=0.3, color='xkcd:orange', label=f'IQR')
a[3][0].set_title('MAE')
a[4][0].set_title('1-SSIM')
a[3][0].legend()
a[4][0].legend()
a[3][0].set_ylim([0,0.035])
a[4][0].set_ylim([0,0.035])

a[0][2].imshow(all_resviz_trad[ssimmedind_trad], cmap='gray', vmin=0, vmax=vminmaxbound)
a[0][3].imshow(all_diffviz_trad[ssimmedind_trad], cmap='seismic', vmin=-1*vminmaxbound, vmax=vminmaxbound)
a[2][2].imshow(all_reforesviz_trad[ssimmedind_trad], cmap='gray', vmin=0, vmax=vminmaxbound, aspect=mpr_plotratio)
a[2][3].imshow(all_refodiffviz_trad[ssimmedind_trad], cmap='seismic', vmin=-1*vminmaxbound, vmax=vminmaxbound, aspect=mpr_plotratio)
a[1][2].imshow(all_resviz_mpr[ssimmedind_mpr], cmap='gray', vmin=0, vmax=vminmaxbound, aspect=mpr_plotratio)
a[1][3].imshow(all_diffviz_mpr[ssimmedind_mpr], cmap='seismic', vmin=-1*vminmaxbound, vmax=vminmaxbound, aspect=mpr_plotratio)
a[0][2].text(140, 20, f'MAE: {allmae_trad[ssimmedind_trad,-1]:.4f}', c='w', size='medium')
a[0][3].text(140, 20, f'MAE: {allmae_trad[ssimmedind_trad,-1]:.4f}', size='medium')
a[0][2].text(120, 35, f'1-SSIM: {allssim_trad[ssimmedind_trad,-1]:.4f}', c='w', size='medium')
a[0][3].text(120, 35, f'1-SSIM: {allssim_trad[ssimmedind_trad,-1]:.4f}', size='medium')

a[2][2].text(10, 10, f'MAE: {allmae_trad[ssimmedind_trad,-1]:.4f}', c='w', size='medium')
a[2][3].text(10, 10, f'MAE: {allmae_trad[ssimmedind_trad,-1]:.4f}', size='medium')
a[2][2].text(10, 95, f'1-SSIM: {allssim_trad[ssimmedind_trad,-1]:.4f}', c='w', size='medium')
a[2][3].text(10, 95, f'1-SSIM: {allssim_trad[ssimmedind_trad,-1]:.4f}', size='medium')

a[1][2].text(10, 10, f'MAE: {allmae_mpr[ssimmedind_mpr,-1]:.4f}', c='w', size='medium')
a[1][3].text(10, 10, f'MAE: {allmae_mpr[ssimmedind_mpr,-1]:.4f}', size='medium')
a[1][2].text(10, 95, f'1-SSIM: {allssim_mpr[ssimmedind_mpr,-1]:.4f}', c='w', size='medium')
a[1][3].text(10, 95, f'1-SSIM: {allssim_mpr[ssimmedind_mpr,-1]:.4f}', size='medium')

plt.savefig('/tmp/debug_plot.pdf')
plt.savefig('./imspace_vs_geo_comparison.png')
plt.show()

