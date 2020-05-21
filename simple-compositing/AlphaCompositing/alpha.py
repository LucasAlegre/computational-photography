from skimage import img_as_float, img_as_ubyte
from skimage.io import imshow, imsave, imread
from scipy import io
import matplotlib.pyplot as plt


def alpha_compositing(fg, bg, alpha):
    fg = img_as_float(fg)
    bg = img_as_float(bg)
    alpha = img_as_float(alpha)
    return img_as_ubyte(alpha * fg + (1 - alpha) * bg)


if __name__ == '__main__':

    bg = imread('background.png')
    fg = imread('GT04.png')
    alpha = imread('GT04_alpha.png')

    result = alpha_compositing(fg, bg, alpha)
    imshow(result)
    plt.show()

    imsave("result_board.png", result)
