#%% Lucas N. Alegre - Discrete Fourier Transform
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from skimage import img_as_float, img_as_ubyte
from skimage.io import imshow, imsave, imread, show


def dft(f):
    F = np.zeros(f.shape, dtype=np.complex128)
    M, N = f.shape
    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    F[u,v] += f[x,y] * np.exp(-1j*2*np.pi*((u*x/M)+(v*y)/N))
    return F

def idft(F):
    f = np.zeros(F.shape, dtype=np.complex128)
    M, N = F.shape
    for x in range(M):
        for y in range(N):
            for u in range(M):
                for v in range(N):
                    f[x,y] += F[u,v] * np.exp(1j*2*np.pi*((u*x/M)+(v*y)/N))
    f *= 1/(M*N)
    return f

def min_max_scale(f):
    return (f - f.min()) / (f.max() - f.min())

def plot_spectrum(f):
    o = np.abs(f)
    o = np.log(o + 1) # avoid log(0)
    o = min_max_scale(o) # normalize to [0,1]
    o = img_as_ubyte(o)
    imshow(o)
    show()
    return o

#%% Task 1 - IDFT reconstruction using only real or imaginary components

img = imread('cameraman.tif')

ft = fft2(img)
real_part = ft.real
imaginary_part = ft.imag

real = ifft2(real_part).astype('uint8')
imsave('reconstructed_real.png', real)
imshow(real)
show()

imaginary = ifft2(1j*imaginary_part)
imaginary = np.clip(imaginary.real, 0, 255).astype('uint8')
imsave('reconstructed_imag.png', imaginary)
imshow(imaginary)
show()

#%% Task 2

img = imread('cameraman_small.tif')
imsave('cameraman_small.png', img)

# a) Scipy DFT -> My IDFT
scipy_dft = fft2(img)
reconstructed = idft(scipy_dft).real.astype('uint8')
imsave('2a.png', reconstructed)
imshow(reconstructed)
show()

# b) My DFT -> Scipy IDFT
my_dft = dft(img)
reconstructed = ifft2(my_dft).real.astype('uint8')
imsave('2b.png', reconstructed)
imshow(reconstructed)
show()

# c) My DFT -> My IDFT
reconstructed = idft(my_dft).real.astype('uint8')
imsave('2c.png', reconstructed)
imshow(reconstructed)
show()

#%% Task 3

img = imread('cameraman.tif')

# a)
a = fft2(img)
a[0,0] = 0
reconstructed = np.clip(ifft2(a).real, 0, 255).astype('uint8')
imsave("dc0.png", reconstructed)
imshow(reconstructed)
show()
#b)
b = np.clip(img - img.mean(), 0, 255).astype('uint8')
imsave('mean_subtract.png', b)
imshow(b)
show()

#%% Task 4
img = imread('cameraman.tif')

# a)
a = fft2(img)
spectrum = plot_spectrum(a)
imsave('spectrum.png', spectrum)
shifted = fftshift(a)
shift_spectrum = plot_spectrum(shifted)
imsave('shifted_spectrum.png', shift_spectrum)
reconstructed = ifft2(shifted).real
reconstructed = np.clip(reconstructed, 0, 255).astype('uint8')
imsave('shift_reconstructed.png', reconstructed)
imshow(reconstructed)
show()

# b)
g = np.empty(img.shape)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        g[x,y] = (-1)**(x+y) * img[x,y]
g = np.clip(g, 0, 255).astype('uint8')
imsave('g.png', g)
imshow(g)
show()

# c)
shifted_img = fftshift(img)
imsave('shifted_cameraman.png', shifted_img)
imshow(shifted_img)
show()
c = fft2(shifted_img)
spectrum = plot_spectrum(c)
imsave('spectrum_of_shifted.png', spectrum)

# %% Task 5
img = imread('cameraman.tif')

# a)
ft = fft2(img)
a = np.abs(ft)
reconstructed = ifft2(a)
reconstructed = np.clip(reconstructed, 0, 255).astype('uint8')
imsave('reconstructed_amplitude.png', reconstructed)
imshow(reconstructed)
show()

#b)
b = np.exp(1j*np.angle(ft))
reconstructed = ifft2(b)
reconstructed = min_max_scale(reconstructed.real)
imsave('reconstructed_phase.png', reconstructed)
imshow(reconstructed)
show()
