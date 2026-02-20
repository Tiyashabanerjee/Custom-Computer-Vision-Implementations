import os
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from utils import imread_im2double, save_image, ensure_dir

# Adding the project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

def sobel(img):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
    ky = kx.T
    ix = signal.convolve2d(img, kx, mode='same', boundary='symm')
    iy = signal.convolve2d(img, ky, mode='same', boundary='symm')
    return ix, iy

def gaussian(sigma, size=7):
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2+yy**2)/(2*sigma*sigma))
    return g/g.sum()

def harris_response(img, k=0.04, sigma=1.5):
    ix, iy = sobel(img)
    ixx = ix*ix
    iyy = iy*iy
    ixy = ix*iy
    g = gaussian(sigma, size=7)
    sxx = signal.convolve2d(ixx, g, mode='same', boundary='symm')
    syy = signal.convolve2d(iyy, g, mode='same', boundary='symm')
    sxy = signal.convolve2d(ixy, g, mode='same', boundary='symm')
    det = sxx*syy - sxy*sxy
    trace = sxx + syy
    R = det - k*(trace**2)
    return R

def nonmax_suppress(R, radius=3, thresh_rel=0.01):
    from scipy.ndimage import maximum_filter
    Rmax = maximum_filter(R, footprint=np.ones((2*radius+1, 2*radius+1)))
    peaks = (R==Rmax) & (R > (R.max()*thresh_rel))
    pts = np.argwhere(peaks)
   
    vals = [R[p[0],p[1]] for p in pts]
    order = np.argsort(vals)[::-1]
    return pts[order]

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    img_path = os.path.join(root, 'Data', 'Question2-1.jpg')
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs'); ensure_dir(out_dir)
    img = imread_im2double(img_path, as_gray=True)
    R = harris_response(img)
    pts = nonmax_suppress(R, radius=4, thresh_rel=0.02)
   
    topk = min(200, len(pts))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    for p in pts[:topk]:
        ax.plot(p[1], p[0], 'ro', markersize=3)
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'q2_harris_corners.png'), bbox_inches='tight')
    plt.close(fig)
    
    normR = (R - R.min())/(R.max()-R.min()+1e-12)
    save_image(normR, os.path.join(out_dir, 'q2_response.png'))
    print("q2 done. Outputs in", out_dir)

if __name__ == '__main__':
    main()