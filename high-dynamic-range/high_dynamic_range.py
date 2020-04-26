from skimage import img_as_float, img_as_ubyte
from skimage.io import imshow, imsave, imread
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


def gamma_decoding(img):
    gamma = 2.2
    decoded = img_as_float(img)**gamma
    return img_as_ubyte(decoded)

def gamma_encoding(img):
    gamma = 1/2.2
    encoded = img_as_float(img)**gamma
    return img_as_ubyte(encoded)

def get_hdr(ldr_images, response_curve, exposure_times):
    hdr = np.zeros(ldr_images[0].shape)
    for k in range(len(ldr_images)):
        image_k = ldr_images[k]
        for i in range(image_k.shape[0]):
            for j in range(image_k.shape[1]):
                for c in range(3):
                    hdr[i,j,c] += response_curve[image_k[i,j,c]][c] / exposure_times[k]

    hdr /= len(ldr_images)

    return hdr

def tonemap_global(img, alpha=0.18):
    luminance = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    avg_luminance = np.exp(np.mean(np.log(luminance + 1e-8)))
    gamma = 1/2.2
    scaled_luminance = (alpha/avg_luminance) * luminance
    
    global_operator = scaled_luminance / (1 + scaled_luminance)

    tonemapped = np.empty(img.shape)
    tonemapped[:,:,0] = global_operator * (img[:,:,0]/luminance)**gamma
    tonemapped[:,:,1] = global_operator * (img[:,:,1]/luminance)**gamma
    tonemapped[:,:,2] = global_operator * (img[:,:,2]/luminance)**gamma

    tonemapped[tonemapped > 1] = 1
    tonemapped *= 255

    return tonemapped.astype('uint8')


if __name__ == '__main__':

    ldr_images = [gamma_decoding(imread(f)) for f in sorted(glob.glob('office/office_*'))]  # Apply gamma decoding to linearilize values
    exposure_times = pd.read_csv('exposure_times.csv')['exposure_time'].values
    response_curve = np.exp(np.loadtxt('curve.txt'))  # Apply exp as the values are stored in log scale

    # HDR
    hdr = get_hdr(ldr_images, response_curve, exposure_times)
    io.savemat('hdr.mat', {'hdr':hdr})  # So I can open in MATLAB

    # Global Tonemmaping
    alpha = 0.18
    tonemapped = tonemap_global(hdr, alpha=alpha)
    imsave('tonemapped_{}.png'.format(alpha), tonemapped)
    alpha = 0.36
    tonemapped = tonemap_global(hdr, alpha=alpha)
    imsave('tonemapped_{}.png'.format(alpha), tonemapped)
    alpha = 0.72
    tonemapped = tonemap_global(hdr, alpha=alpha)
    imsave('tonemapped_{}.png'.format(alpha), tonemapped)
