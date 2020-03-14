# %%
import rawpy
import imageio
import numpy as np
from numpy.matlib import repmat
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def im2double(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out

def im2uint8(img):
    return (img*255).astype('uint8')

def demosaic(cfa):
    height, width = cfa.shape
    out = np.zeros((height, width, 3))

    r_bayer = np.array([[1, 0],
                        [0, 0]])
    g_bayer = np.array([[0, 1], 
                        [1, 0]])
    b_bayer = np.array([[0, 0], 
                        [0, 1]])

    R = np.multiply(cfa, repmat(r_bayer, height//2, width//2))
    G = np.multiply(cfa, repmat(g_bayer, height//2, width//2))
    B = np.multiply(cfa, repmat(b_bayer, height//2, width//2))

    diamond = np.array([[0, 1/4, 0],
			            [1/4, 0, 1/4], 
			            [0, 1/4, 0]])

    corners = np.array([[1/4, 0, 1/4],
			            [0, 0, 0], 
			            [1/4, 0, 1/4]])
    
    R += convolve(R, corners)
    out[:,:,0] = R + convolve(R, diamond)

    G += convolve(G, diamond)
    out[:,:,1] = G

    B += convolve(B, corners)
    out[:,:,2] = B + convolve(B, diamond)

    return out

def white_balance(img, wb_pixel):
    r_wb = img[wb_pixel[0], wb_pixel[1], 0]
    g_wb = img[wb_pixel[0], wb_pixel[1], 1]
    b_wb = img[wb_pixel[0], wb_pixel[1], 2]

    img[:,:,0] *= 1./r_wb
    img[:,:,1] *= 1./g_wb
    img[:,:,2] *= 1./b_wb

    return img

def gamma_encoding(img, gamma=1.0/2.2):
    return img**gamma

# %%
raw = rawpy.imread('scene_raw.dng').raw_image * 2**4
raw = im2double(raw)
imageio.imwrite('./scene_raw.png', im2uint8(raw))

# %%
image_demosaic = demosaic(raw)
imageio.imwrite('./scene_demosaic.png', im2uint8(image_demosaic))

# %%
image_wb = white_balance(image_demosaic, wb_pixel=(640, 2132))
imageio.imwrite('./scene_whitebalanced.png', im2uint8(image_wb))

# %%
image = gamma_encoding(image_wb, gamma=1/1.8)
plt.imshow(im2uint8(image))
imageio.imwrite('./scene_gammaencoded.png', im2uint8(image))

