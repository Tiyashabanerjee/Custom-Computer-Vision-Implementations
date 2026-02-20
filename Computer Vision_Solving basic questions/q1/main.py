import os
import sys
import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from utils import imread_im2double, save_image, ensure_dir
import glob

# Adding the project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

def sobel_filters(img):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
    ky = kx.T
    ix = signal.convolve2d(img, kx, boundary='symm', mode='same')
    iy = signal.convolve2d(img, ky, boundary='symm', mode='same')
    return ix, iy

def trace_boundary(edge_map):
    # findng the connected components and for each component extracting the boundary pixels and ordered chain
    labeled, n = ndimage.label(edge_map)
    boundaries = []
    for lab in range(1, n+1):
        comp = (labeled == lab)
        if comp.sum() < 10: 
            continue
        # boundary pixels: pixel in comp with at least one background neighbor
        pad = np.pad(comp, 1, mode='constant', constant_values=0)
        coords = np.argwhere(comp)
        bcoords = []
        for (r,c) in coords:
            neighborhood = pad[r:r+3, c:c+3]
            if np.any(neighborhood.flatten()==0):
                bcoords.append((r,c))
        if not bcoords:
            continue
        
        ordered = [bcoords.pop(np.argmin([p[1] for p in bcoords]))]  # start at leftmost
        while bcoords:
            last = ordered[-1]
            dists = [ ( (p[0]-last[0])**2 + (p[1]-last[1])**2, i ) for i,p in enumerate(bcoords) ]
            dists.sort()
            _, idx = dists[0]
            ordered.append(bcoords.pop(idx))
            if len(ordered)>10000:
                break
        boundaries.append(np.array(ordered))
    return boundaries

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_path = "Data\\Question3\\Question1.png"  # existing assignment
    # I am adding this fallback: if image is not found, to try and locate it under project Data/
    if not os.path.exists(img_path):
        basename = os.path.basename(img_path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        matches = glob.glob(os.path.join(project_root, '**', basename), recursive=True)
        if matches:
            img_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Image not found: '{img_path}'. Place the file under {os.path.join(project_root, 'Data')} "
                "or update the path in q1/main.py."
            )

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs'); ensure_dir(out_dir)
    img = imread_im2double(img_path, as_gray=True)
    sigma = 1.4
    gk = gaussian_kernel(7, sigma)
    sm = signal.convolve2d(img, gk, mode='same', boundary='symm')
    ix, iy = sobel_filters(sm)
    grad = np.sqrt(ix**2 + iy**2)
    ori = np.arctan2(iy, ix)

    # saving gradient magnitude
    save_image((grad/grad.max()), os.path.join(out_dir, 'q1_gradient_magnitude.png'))

    # quiver overlay plot 
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    step = max(1, img.shape[1]//60)
    Y, X = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
    U = ix[::step, ::step]; V = -iy[::step, ::step]
    ax.quiver(X, Y, U, V, scale=50)
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'q1_quiver_overlay.png'), bbox_inches='tight')
    plt.close(fig)

    # Edge map
    thresh = np.percentile(grad, 75)
    edge_map = grad > thresh
    save_image(edge_map.astype(float), os.path.join(out_dir, 'q1_edge_map.png'))

    # Boundary tracing
    boundaries = trace_boundary(edge_map)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    for b in boundaries:
        if b.shape[0] > 1:
            ax.plot(b[:,1], b[:,0], '-r', linewidth=1)
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'q1_boundaries.png'), bbox_inches='tight')
    plt.close(fig)
    print("q1 done. Outputs in", out_dir)

if __name__ == '__main__':
    main()