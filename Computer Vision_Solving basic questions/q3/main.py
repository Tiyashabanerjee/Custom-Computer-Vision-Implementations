import os
import sys
import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from utils import imread_im2double, save_image, ensure_dir

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

def myHoughLine(imBW, n=5, thetas=None):
    if thetas is None:
        thetas = np.deg2rad(np.arange(-90,90))
    h, w = imBW.shape
    diag = int(np.ceil(np.hypot(h,w)))
    rhos = np.linspace(-diag, diag, 2*diag+1)
    A = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(imBW)
    cos_t = np.cos(thetas); sin_t = np.sin(thetas)
    for (x,y) in zip(xs, ys):
        r = x * cos_t + y * sin_t
        idx = np.round(r).astype(int) + diag
        for ti, ri in enumerate(idx):
            A[ri, ti] += 1
    lines = []
    A_copy = A.copy()
    for _ in range(n):
        idx = np.unravel_index(np.argmax(A_copy), A_copy.shape)
        votes = A_copy[idx]
        if votes == 0:
            break
        rho = rhos[idx[0]]; theta = thetas[idx[1]]
        lines.append((rho, theta, votes))
        # suppressing neighborhood
        r0, t0 = idx
        rr = slice(max(0,r0-10), min(A_copy.shape[0], r0+11))
        tt = slice(max(0,t0-10), min(A_copy.shape[1], t0+11))
        A_copy[rr, tt] = 0
    return lines

def draw_lines(img, lines, outpath):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    h,w = img.shape
    for rho,theta,_ in lines:
        a = np.cos(theta); b = np.sin(theta)
        x0 = a*rho; y0 = b*rho
        x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
        ax.plot([x1,x2],[y1,y2],'-r')
    ax.set_axis_off()
    fig.savefig(outpath, bbox_inches='tight'); plt.close(fig)

# Circle Hough 
def myHoughCircleTrain(imBW, c, ptlist):
    
    pts = np.array(ptlist)
    rads = np.sqrt(((pts - np.array(c))**2).sum(axis=1))
    r = int(np.round(rads.mean()))
    return {'r': r}

def myHoughCircleTest(imBWnew, cellvar, topk=2):
    r = cellvar['r']
    h,w = imBWnew.shape
    acc = np.zeros((h,w), dtype=np.int32)
    ys, xs = np.nonzero(imBWnew)
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    for (x,y) in zip(xs, ys):
        for ang in angles:
            cx = int(round(x - r*np.cos(ang)))
            cy = int(round(y - r*np.sin(ang)))
            if 0 <= cy < h and 0 <= cx < w:
                acc[cy, cx] += 1
    centers = []
    accc = acc.copy()
    for _ in range(topk):
        idx = np.unravel_index(np.argmax(accc), accc.shape)
        if accc[idx] == 0:
            break
        centers.append((idx[1], idx[0], accc[idx]))  
        # suppressing neighborhood
        rr = slice(max(0, idx[0]-r), min(h, idx[0]+r+1))
        cc = slice(max(0, idx[1]-r), min(w, idx[1]+r+1))
        accc[rr, cc] = 0
    return centers

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    in_path = os.path.join(root, 'Data', 'Question3', 'Question3', '3.png')
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs'); ensure_dir(out_dir)
    img = imread_im2double(in_path, as_gray=True)
    # simple edge: sobel magnitude threshold
    sx = ndimage.sobel(img, axis=1)
    sy = ndimage.sobel(img, axis=0)
    mag = np.hypot(sx, sy)
    edge = mag > np.percentile(mag, 75)
    save_image(edge.astype(float), os.path.join(out_dir, 'q3_edge.png'))

    # Hough lines
    lines = myHoughLine(edge, n=5)
    draw_lines(img, lines, os.path.join(out_dir, 'q3_lines.png'))

    train_path = os.path.join(root, 'Data', 'Question3', 'Question3', 'train.png')
    train = imread_im2double(train_path, as_gray=True)
    # assuming center roughly center of image and boundary points are edge pixels of train
    sx2 = ndimage.sobel(train, axis=1); sy2 = ndimage.sobel(train, axis=0)
    mag2 = np.hypot(sx2, sy2); edge_train = mag2 > np.percentile(mag2, 75)
    ys, xs = np.nonzero(edge_train)
    
    c = (train.shape[0]//2, train.shape[1]//2)
    ptlist = list(zip(ys, xs))
    cell = myHoughCircleTrain(edge_train, c, ptlist)
    centers = myHoughCircleTest(edge, cell, topk=2)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap='gray')
    for (x,y,v) in centers:
        circ = plt.Circle((x,y), cell['r'], fill=False, color='r', linewidth=2)
        ax.add_patch(circ)
        ax.plot(x,y,'ro')
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, 'q3_circles.png'), bbox_inches='tight')
    plt.close(fig)
    print("q3 done. Outputs in", out_dir)

if __name__ == '__main__':
    main()
