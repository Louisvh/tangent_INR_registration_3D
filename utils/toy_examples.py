import numpy as np
import os
import torch
import SimpleITK as sitk
from scipy import ndimage 
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_toy_func(ex_id, shape, nsams=500, max_t=1.8):
    t = np.linspace(0,max_t,nsams)
    if ex_id == 'S':
        x = (t-max_t/2)**1 / 4 + 0.5
        y = (t-max_t/2)**5 / 2 + 0.5
    elif ex_id == '-':
        x = t/max_t*0.7+0.15
        y = t*0 + 0.5 
    else:
        print(f'ex_id {ex_id} not implemented')
        
    cx = [shape[2]*c for c in x]
    cy = [shape[1]*c for c in y]
    dt = [t[i] - t[i-1] for i in range(1,nsams)]
    dx = [cx[i] - cx[i-1] for i in range(1,nsams)]
    dy = [cy[i] - cy[i-1] for i in range(1,nsams)]
    dnorm = [np.linalg.norm((dx[i],dy[i])) for i in range(len(dx))]

    arclen = np.sum(dnorm)
    #print(f'arclen: {arclen}')
    t_resam = [0]
    for i in range(nsams-1):
        relnorm = (dnorm[i] / arclen) / (1/(nsams-1)) 
        dt_resam = dt[i] / relnorm  
        t_resam.append(t_resam[-1]+dt_resam)

    t_resam = np.array(t_resam)
    t_resam *= max_t / np.max(t_resam)
    if ex_id == 'S':
        x = (t_resam-max_t/2)**1 / 4 + 0.5
        y = (t_resam-max_t/2)**5 / 2 + 0.5
    elif ex_id == '-':
        x = t_resam/max_t*0.7+0.15
        y = t_resam*0 + 0.5 
    else:
        print(f'ex_id {ex_id} not implemented')
    cx = [shape[2]*c for c in x]
    cy = [shape[1]*c for c in y]
    dx = [cx[i] - cx[i-1] for i in range(1,nsams)]
    dy = [cy[i] - cy[i-1] for i in range(1,nsams)]
    dnorm = [np.linalg.norm((dx[i],dy[i])) for i in range(len(dx))]
    dx.insert(0, dx[0])
    dy.insert(0, dy[0]) 
    dnorm.insert(0, dnorm[0])

    return t_resam, cx, cy, dx, dy, dnorm

def paint_sphere(canvas, coord, r):
    cz, cy, cx = coord
    zrange = (max(int(cz)-int(np.round(r+0.5)),0), 
              min(int(cz)+int(np.round(r+1.5)),canvas.shape[0]))
    yrange = (max(int(cy)-int(np.round(r+0.5)),0), 
              min(int(cy)+int(np.round(r+1.5)),canvas.shape[1]))
    xrange = (max(int(cx)-int(np.round(r+0.5)),0), 
              min(int(cx)+int(np.round(r+1.5)),canvas.shape[2]))
    allz = [z for z in range(*zrange)]
    ally = [y for y in range(*yrange)]
    allx = [x for x in range(*xrange)]
    mz,my,mx = np.meshgrid(allz,ally,allx)
    alldistcoords = np.stack((mz-cz,my-cy,mx-cx))
    alldists = np.linalg.norm(alldistcoords,axis=0)
    all_reldists = alldists / r
    all_invreldists = 1-all_reldists
    mblock = canvas[zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]] 
    combiblock = np.stack([mblock, all_invreldists])
    newblock = np.max(combiblock, axis=0)
    canvas[zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]] = newblock
    
    return canvas
 
def load_simple_toy_example(ex_id, shape=(3,256,256), widths=(5,8), maskwidth=32, subfac=1, max_t=2, b_delay=0.1, ret_images=True):
    voxel_size = (1,1,1)
    
    canvasA = np.zeros(shape)
    canvasB = np.zeros(shape)
    nsams=50
    subfac2 = 0.25
    t_resam, cx, cy, dx, dy, dnorm = get_toy_func(ex_id, shape, nsams, max_t=max_t)
    cz = [shape[0]/2 for _ in cx]
    s = widths[0] / widths[1]
    n_dips = 2
    squeezewidths_a = [widths[1] * (s+(1-s)*np.sin(t)**2) 
                        for t in np.linspace(np.pi/2,np.pi*(n_dips+0.5), nsams)]
    squeezewidths_b = [widths[1] * (s+(1-s)*np.sin(t+b_delay)**2) 
                        for t in np.linspace(np.pi/2,np.pi*(n_dips+0.5), nsams)]
    
    for i in tqdm(range(nsams)):
        linecoord = (cz[i],cy[i],cx[i])
        canvasA = paint_sphere(canvasA, linecoord, squeezewidths_a[i])
        
    for i in tqdm(range(nsams)):
        linecoord = (cz[i],cy[i],cx[i])
        canvasB = paint_sphere(canvasB, linecoord, squeezewidths_b[i])

    cv_foreground = (canvasA+canvasB > 0) * 1
    mask = ndimage.binary_dilation(cv_foreground, iterations=maskwidth-widths[1])
    cv_fgdist = ndimage.distance_transform_edt(1-cv_foreground)
    cv_fgdist /= np.max(cv_fgdist)
    cv_fgdist = (1 - cv_fgdist) / 10
    for canvas in [canvasA, canvasB]:
        canvas += cv_fgdist
    if shape[0] == 3:
        mask[0::2,:,:] = 0

    return torch.as_tensor(canvasA).float(), torch.as_tensor(canvasB).float(), voxel_size, mask

   

def load_bendy_toy_example(ex_id, shape=(3,256,256), widths=(5,8), maskwidth=32, subfac=1, max_t=2, ret_images=True):
    voxel_size = (1,1,1)
    
    canvasA = np.zeros(shape)
    canvasB = np.zeros(shape)
    nsams=50
    subfac2 = 0.25
    t_resam, cx, cy, dx, dy, dnorm = get_toy_func(ex_id, shape, nsams, max_t=max_t)
    cz = [shape[0]/2 for _ in cx]
    basewidths = [widths[1] for t in t_resam]
    s = widths[0] / widths[1]
    n_dips = 2
    squeezewidths = [widths[1] * (s+(1-s)*np.sin(t)**2) 
                        for t in np.linspace(np.pi/2,np.pi*(n_dips+0.5), nsams)]
    
    for i in tqdm(range(nsams)):
        linecoord = (cz[i],cy[i],cx[i])
        if ret_images:
            canvasA = paint_sphere(canvasA, linecoord, basewidths[i])
        
    for i in tqdm(range(nsams)):
        linecoord = (cz[i],cy[i],cx[i])
        if ret_images:
            canvasB = paint_sphere(canvasB, linecoord, squeezewidths[i])

    imAside1x = [cx[i]+subfac*dy[i]/dnorm[i]*basewidths[i] for i in range(len(cx))]
    imAside1y = [cy[i]-subfac*dx[i]/dnorm[i]*basewidths[i] for i in range(len(cy))]
    imAside2x = [cx[i]-subfac*dy[i]/dnorm[i]*basewidths[i] for i in range(len(cx))]
    imAside2y = [cy[i]+subfac*dx[i]/dnorm[i]*basewidths[i] for i in range(len(cy))]
    imBside1x = [cx[i]+subfac*dy[i]/dnorm[i]*squeezewidths[i] for i in range(len(cx))]
    imBside1y = [cy[i]-subfac*dx[i]/dnorm[i]*squeezewidths[i] for i in range(len(cy))]
    imBside2x = [cx[i]-subfac*dy[i]/dnorm[i]*squeezewidths[i] for i in range(len(cx))]
    imBside2y = [cy[i]+subfac*dx[i]/dnorm[i]*squeezewidths[i] for i in range(len(cy))]

    if False:
        n_end_b = int(np.linalg.norm((imAside1x[-1]-imAside2x[-1],imAside1y[-1]-imAside2y[-1]))/dnorm[-1])
        n_end_f = int(np.linalg.norm((imAside1x[0]-imAside2x[0],imAside1y[0]-imAside2y[0]))/dnorm[0])
        n_end_b = int(np.linalg.norm((imBside1x[-1]-imBside2x[-1],imBside1y[-1]-imBside2y[-1]))/dnorm[-1])
        n_end_f = int(np.linalg.norm((imBside1x[0]-imBside2x[0],imBside1y[0]-imBside2y[0]))/dnorm[0])
    else:
        n_end_b = 4
        n_end_f = 4

    endsAx1 = [imAside1x[-1]*c + imAside2x[-1]*(1-c) for c in np.linspace(0,1,n_end_b)]
    endsAx2 = [imAside1x[0]*c + imAside2x[0]*(1-c) for c in np.linspace(0,1,n_end_f)]
    endsBx1 = [imBside1x[-1]*c + imBside2x[-1]*(1-c) for c in np.linspace(0,1,n_end_b)]
    endsBx2 = [imBside1x[0]*c + imBside2x[0]*(1-c) for c in np.linspace(0,1,n_end_f)]
    endsAy1 = [imAside1y[-1]*c + imAside2y[-1]*(1-c) for c in np.linspace(0,1,n_end_b)]
    endsAy2 = [imAside1y[0]*c + imAside2y[0]*(1-c) for c in np.linspace(0,1,n_end_f)]
    endsBy1 = [imBside1y[-1]*c + imBside2y[-1]*(1-c) for c in np.linspace(0,1,n_end_b)]
    endsBy2 = [imBside1y[0]*c + imBside2y[0]*(1-c) for c in np.linspace(0,1,n_end_f)]
    allAx = imAside1x + endsAx1 + imAside2x + endsAx2
    allBx = imBside1x + endsBx1 + imBside2x + endsBx2
    allAy = imAside1y + endsAy1 + imAside2y + endsAy2
    allBy = imBside1y + endsBy1 + imBside2y + endsBy2
    all_squeezewidths = squeezewidths + [squeezewidths[-1] for x in endsAx1]
    all_squeezewidths += squeezewidths + [squeezewidths[0] for x in endsAx2]

    lmss = 1
    lmA = np.asarray([(shape[0]/2, allAy[i], allAx[i]) for i in range(0,len(allAx),lmss)])
    lmB = np.asarray([(shape[0]/2, allBy[i], allBx[i]) for i in range(0,len(allBx),lmss)])
    all_squeezewidths = np.asarray(all_squeezewidths[::lmss])

    if ret_images:
        cv_foreground = (canvasA > 0) * 1
        mask = ndimage.binary_dilation(cv_foreground, iterations=maskwidth-widths[1])
        cv_fgdist = ndimage.distance_transform_edt(1-cv_foreground)
        cv_fgdist /= np.max(cv_fgdist)
        cv_fgdist = (1 - cv_fgdist) / 10
        for canvas in [canvasA, canvasB]:
            canvas += cv_fgdist
        if shape[0] == 3:
            mask[0::2,:,:] = 0

        return torch.as_tensor(canvasA).float(), torch.as_tensor(canvasB).float(), lmA, lmB, mask, voxel_size, all_squeezewidths
    else:
        return lmA, lmB

    


def load_uniform_toy_example(ex_id, shape=(3,256,256), widths=(5,8), maskwidth=32):
    voxel_size = (1,1,1)
    
    canvas = np.ones(shape)
    if ex_id == 'S':
        nsams = 500
        max_t = 1.8
        t = np.linspace(0,max_t,nsams)
        x = (t-max_t/2)**1 / 4 + 0.5
        y = (t-max_t/2)**5 / 2 + 0.5
        cx = [shape[2]*c for c in x]
        cy = [shape[1]*c for c in y]
        dt = [t[i] - t[i-1] for i in range(1,nsams)]
        dx = [cx[i] - cx[i-1] for i in range(1,nsams)]
        dy = [cy[i] - cy[i-1] for i in range(1,nsams)]
        dnorm = [np.linalg.norm((dx[i],dy[i])) for i in range(len(dx))]

        arclen = np.sum(dnorm)
        #print(f'arclen: {arclen}')
        t_resam = [0]
        for i in range(nsams-1):
            relnorm = (dnorm[i] / arclen) / (1/(nsams-1)) 
            dt_resam = dt[i] / relnorm  
            t_resam.append(t_resam[-1]+dt_resam)

        t_resam = np.array(t_resam)
        t_resam *= max_t / np.max(t_resam)
        x = (t_resam-max_t/2)**1 / 4 + 0.5
        y = (t_resam-max_t/2)**5 / 2 + 0.5
        cx = [shape[2]*c for c in x]
        cy = [shape[1]*c for c in y]
        dx = [cx[i] - cx[i-1] for i in range(1,nsams)]
        dy = [cy[i] - cy[i-1] for i in range(1,nsams)]
        dnorm = [np.linalg.norm((dx[i],dy[i])) for i in range(len(dx))]

        discx = [int(np.round(x)) for x in cx]
        discy = [int(np.round(y)) for y in cy]

        canvas[shape[0]//2,discy,discx] = 0
        nskip = 15
        for i in range(50,len(discy),50):
            canvas[shape[0]//2,discy[i:i+nskip],discx[i:i+nskip]] = 1
        #from matplotlib import pyplot as plt
        #plt.imshow(canvas[1,:].T)
        #plt.show()

        lmA = []
        lmB = []
        for i in range(25,nsams-20,15):
            lmA.append((shape[0]/2, cy[i]+dx[i]/dnorm[i]*widths[0], cx[i]-dy[i]/dnorm[i]*widths[0]))
            lmB.append((shape[0]/2, cy[i]+dx[i]/dnorm[i]*widths[1], cx[i]-dy[i]/dnorm[i]*widths[1]))
            lmA.append((shape[0]/2, cy[i]-dx[i]/dnorm[i]*widths[0], cx[i]+dy[i]/dnorm[i]*widths[0]))
            lmB.append((shape[0]/2, cy[i]-dx[i]/dnorm[i]*widths[1], cx[i]+dy[i]/dnorm[i]*widths[1]))
        lmA.append((shape[0]/2, cy[11], cx[11]))
        lmA.append((shape[0]/2, cy[-12], cx[-12]))
        lmA.append((shape[0]/2, discy[0]-dy[0]/dnorm[0]*widths[0], discx[0]-dx[0]/dnorm[0]*widths[0]))
        lmA.append((shape[0]/2, discy[-1]+dy[-1]/dnorm[-1]*widths[0], discx[-1]+dx[-1]/dnorm[-1]*widths[0]))
        lmB.append((shape[0]/2, cy[11], cx[11]))
        lmB.append((shape[0]/2, cy[-12], cx[-12]))
        lmB.append((shape[0]/2, discy[0]-dy[0]/dnorm[0]*widths[1], discx[0]-dx[0]/dnorm[0]*widths[1]))
        lmB.append((shape[0]/2, discy[-1]+dy[-1]/dnorm[-1]*widths[1], discx[-1]+dx[-1]/dnorm[-1]*widths[1]))

    elif ex_id == '-':
        edge = 50
        r_e = edge/shape[2] + 0.05
        canvas[shape[0]//2,shape[1]//2,edge:-edge] = 0 
        llen = shape[2] - 2*edge

        nskip = 10
        for i in range(1,10):
            canvas[shape[0]//2,shape[1]//2,int(edge+llen/10*i):int(edge+llen/10*i)+nskip] = 1 
        
        lmA  = [(shape[0]/2,shape[1]//2-widths[0],c*shape[2]) for c in np.linspace(r_e,1-r_e,30)]
        lmA += [(shape[0]/2,shape[1]//2+widths[0],c*shape[2]) for c in np.linspace(r_e,1-r_e,30)]
        lmB  = [(shape[0]/2,shape[1]//2-widths[1],c*shape[2]) for c in np.linspace(r_e,1-r_e,30)]
        lmB += [(shape[0]/2,shape[1]//2+widths[1],c*shape[2]) for c in np.linspace(r_e,1-r_e,30)]
        lmA += [(shape[0]/2,shape[1]//2,edge+5),
                (shape[0]/2,shape[1]//2,shape[2]-edge-5)]
        lmB += [(shape[0]/2,shape[1]//2,edge+5),
                (shape[0]/2,shape[1]//2,shape[2]-edge-5)]
        lmA += [(shape[0]/2,shape[1]//2,edge-widths[0]),
                (shape[0]/2,shape[1]//2,shape[2]-edge+widths[0])]
        lmB += [(shape[0]/2,shape[1]//2,edge-widths[1]),
                (shape[0]/2,shape[1]//2,shape[2]-edge+widths[1])]
    else:
        print(f'ex_id {ex_id} not implemented')

    distcv = ndimage.distance_transform_edt(canvas)
    canvasses = [distcv.copy(), distcv.copy()]
    
    for i in range(len(widths)):
        canvasses[i] = widths[i] - np.clip(canvasses[i], 0, widths[i])
        canvasses[i] /= widths[i]
        canvasses[i] = np.sqrt(canvasses[i])
        cv_foreground = canvasses[i] > 0
        cv_fgdist = ndimage.distance_transform_edt(1-cv_foreground)
        cv_fgdist /= np.max(cv_fgdist)
        cv_fgdist = (1 - cv_fgdist) / 10
        canvasses[i] += cv_fgdist

    mask = (distcv < maskwidth) * 1
    if shape[0] == 3:
        mask[0::2,:,:] = 0
    im_A, im_B = canvasses
    lmA = np.asarray(lmA)
    lmB = np.asarray(lmB)

    return torch.as_tensor(im_A), torch.as_tensor(im_B), lmA, lmB, mask, voxel_size

