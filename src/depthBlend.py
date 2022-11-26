import cv2
import numpy as np

def blender(nprImage, grayImage, grayFactor=0.5):
    '''
    Input:
    nprImage - RGB image.
    grayImage - Grayscale image.
    grayFactor - Floating number.

    Output:
    blendedImage - RGB image.

    Blends the grayscale image of the original photo with the flat non photo realistic image. Both images are blurred before the element wise multiplication.
    '''

    blendedImage = cv2.GaussianBlur(nprImage, (3,3), 0.1)
    blendedImage = blendedImage.astype(np.float64) / 255.0
    colorDepth = cv2.GaussianBlur(grayImage, (3,3), 5)
    colorDepth = (colorDepth.astype(np.float64) * grayFactor) / 255.0
    colorDepth = colorDepth + (0.5 - np.mean(colorDepth))

    blendedImage[:,:,0] = blendedImage[:, :, 0] * colorDepth
    blendedImage[:,:,1] = blendedImage[:, :, 1] * colorDepth
    blendedImage[:,:,2] = blendedImage[:, :, 2] * colorDepth
    blendedImage = blendedImage / np.max(blendedImage)
    blendedImage = np.clip((blendedImage * 255.0), 0, 255).astype(np.uint8)

    return blendedImage