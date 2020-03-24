# %% Author: Lucas N. Alegre - Computational Photography 2020/1 UFRGS - Prof. Manuel Menezes de Oliveira Neto
import rawpy
import imageio
import numpy as np
from numpy.matlib import repmat
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def im2double(img): 
    """Converts image to double in range [0, 1],
    equivalent to matlab's im2double
    
    Args:
        img (np.ndarray): Image
    Returns:
        np.ndarray: The image in range [0, 1]
    """
    min_val = 0
    max_val = np.max(img.ravel())
    if max_val < 255:
        max_val = 255
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out

def im2uint8(img):
    """Converts image to uint8 in range [0, 255],
    equivalent to matlab's im2uint8
    
    Args:
        img (np.ndarray): Image
    Returns:
        np.ndarray: The image in range [0, 255]
    """  
    return (img*255).astype('uint8')

def demosaic(cfa):
    """Bilinear demosaic of Bayer Color Filter Array
    
    Args:
        cfa (np.ndarray): Bayer color filter array
    Returns:
        np.ndarray: 3d array of RGB image
    """    
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

    return np.clip(out, 0.0, 1.0)

def white_balance(img, method='grayworld', wb_pixel=None):
    """White balance of RGB image
    Args:
        img (np.ndarray): Image
        method (str, optional): Method of whitebalance, {'user', 'grayworld'}. Defaults to 'grayworld'.
        wb_pixel (tuple, optional): The reference pixel in case method is 'user'. Defaults to None.
    
    Returns:
        np.ndarray: Image white balanced
    """    
    if method == 'grayworld':
        r_wb = np.mean(img[:,:,0])
        g_wb = np.mean(img[:,:,1])
        b_wb = np.mean(img[:,:,2])
        alpha = g_wb/r_wb
        beta = g_wb/b_wb

        img[:,:,0] *= alpha
        img[:,:,2] *= beta

    elif method == 'user' and wb_pixel is not None:
        r_wb = img[wb_pixel[0], wb_pixel[1], 0]
        g_wb = img[wb_pixel[0], wb_pixel[1], 1]
        b_wb = img[wb_pixel[0], wb_pixel[1], 2]

        img[:,:,0] *= 1./r_wb
        img[:,:,1] *= 1./g_wb
        img[:,:,2] *= 1./b_wb
        
    else:
        raise('Method not implemented.')

    return np.clip(img, 0.0, 1.0)

def gamma_encoding(img, gamma=1.0/2.2):
    """Gamma encoding
    Args:
        img (np.ndarray): Image
        gamma (float, optional): Gamma value used. Defaults to 1.0/2.2.
    Returns:
        np.ndarray: Image gamma encoded
    """    
    return img**gamma

# %% Read dng file, and convert it to double in range [0, 1]
raw = rawpy.imread('scene_raw.dng').raw_image * 2**4 # Scale values to 16-bit representation
raw = im2double(raw)
imageio.imwrite('./scene_raw.png', im2uint8(raw))

# %% Demosaic image to 3d array of RGB values
image_demosaic = demosaic(raw)
imageio.imwrite('./scene_demosaic.png', im2uint8(image_demosaic))

# %% White balance
icon_px = (640, 2132)
paper_px = (2413, 1691)
adapter_px = (1844, 3027)

image_wb_icon = white_balance(image_demosaic, method='user', wb_pixel=icon_px)
imageio.imwrite('./scene_whitebalanced_{}x{}.png'.format(icon_px[0], icon_px[1]), im2uint8(image_wb_icon))

image_wb_paper = white_balance(image_demosaic, method='user', wb_pixel=paper_px)
imageio.imwrite('./scene_whitebalanced_{}x{}.png'.format(paper_px[0], paper_px[1]), im2uint8(image_wb_paper))

image_wb_adapter = white_balance(image_demosaic, method='user', wb_pixel=adapter_px)
imageio.imwrite('./scene_whitebalanced_{}x{}.png'.format(adapter_px[0], adapter_px[1]), im2uint8(image_wb_adapter))

image_wb_grayworld = white_balance(image_demosaic, method='grayworld')
imageio.imwrite('./scene_whitebalanced_grayworld.png', im2uint8(image_wb_grayworld))

# %% Gamma encoding
image = gamma_encoding(image_wb_grayworld, gamma=1/2.2)
imageio.imwrite('./scene_gammaencoded_22.png', im2uint8(image))

image = gamma_encoding(image_wb_grayworld, gamma=1/1.2)
imageio.imwrite('./scene_gammaencoded_12.png', im2uint8(image))

image = gamma_encoding(image_wb_grayworld, gamma=1/1.7)
imageio.imwrite('./scene_gammaencoded_17.png', im2uint8(image))
