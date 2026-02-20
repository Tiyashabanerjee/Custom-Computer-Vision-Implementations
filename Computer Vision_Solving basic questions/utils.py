import os
import sys
import glob
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_ubyte

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    # so that if path is empty (file in current dir) nothing is done
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def imread_im2double(path, as_gray=False):
    """Read image and convert to float64 in [0,1] range."""
    
    candidates = []
    if os.path.isabs(path):
        candidates = [path]
    else:
        candidates = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(__file__), path),
            os.path.join(os.path.dirname(__file__), '..', path)
        ]

    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break

    if found is None:
        
        basename = os.path.basename(path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, 'Data')
        matches = []
        if os.path.exists(data_dir):
            matches = glob.glob(os.path.join(data_dir, '**', basename), recursive=True)
        if matches:
            found = matches[0]

    if found is None:
        raise FileNotFoundError(
            f"No such file: '{path}'. Tried: {candidates}. "
            f"Also searched Data/ and found: {matches if 'matches' in locals() else []}. "
            "Place the image under Data/ or update the image path in the code."
        )

   
    img = imread(found)
    img = np.asarray(img)

    # ensures float64 for processing
    if as_gray:
       
        if img.ndim == 3 and img.shape[2] in (3, 4):
           
            img = img.astype(np.float64)
            maxv = img.max() if img.size else 1.0
            if maxv != 0:
                img = img / maxv
            img = rgb2gray(img)
        else:
            img = img.astype(np.float64)
    else:
        img = img.astype(np.float64)

    # final normalization to [0,1]
    maxv = img.max() if img.size else 1.0
    if maxv != 0:
        img /= maxv
    return img

def imshow_gray(img, title="Image"):
    """Display grayscale image."""
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, path):
    """Save image to given path."""
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)
    imsave(path, img_as_ubyte(image))
