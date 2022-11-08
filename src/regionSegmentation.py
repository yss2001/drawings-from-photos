import cv2
import numpy as np
import skimage.morphology

def colorPalette(image, structureSize=15, minimumEdge=50):
    '''
    Input:
    image - RGB image.
    structureSize - Size of structuring element used in morphology.
    minimumEdge - Minimum number of pixels to be present for a connected component to be retained as an edge.

    Output:
    paletteImage - RGB image.

    Applies morphological reconstruction on the grayscale image followed by gradient operator to segment it into regions. Uses colors from the LAB space of the color image and assigns the mean value for each region.
    '''

    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    element = skimage.morphology.disk(structureSize)
    eroded = cv2.erode(grayImage, element)
    dilationReconstruct = skimage.morphology.reconstruction(eroded, grayImage).astype(np.uint8)
    dilated = cv2.dilate(dilationReconstruct, element)
    erosionReconstruct = skimage.morphology.reconstruction(np.invert(dilated), np.invert(dilationReconstruct)).astype(np.uint8)
    grayRegions = np.invert(erosionReconstruct).astype(np.uint8)

    gradientThreshold = 10
    sobelX = cv2.Sobel(grayRegions, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(grayRegions, cv2.CV_64F, 0, 1)
    magnitude = (np.sqrt(sobelX ** 2.0 + sobelY ** 2.0) > gradientThreshold).astype(np.uint8)

    count, magnitude, _, _ = cv2.connectedComponentsWithStats(magnitude, 8, cv2.CV_32S)
    magnitude[magnitude > 0] = 1

    for i in range(count):
        if (np.sum(magnitude == i) < minimumEdge):
            magnitude[magnitude == i] = 0

    labelCount, labels, _, _ = cv2.connectedComponentsWithStats((1 - magnitude).astype(np.uint8), 8, cv2.CV_32S)
    paletteImage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    for i in range(1, labelCount):
        colors = paletteImage[labels == i]
        color = np.mean(colors, axis=0)
        paletteImage[labels == i] = color
    
    paletteImage = cv2.cvtColor(paletteImage, cv2.COLOR_LAB2RGB)
    paletteImage = (paletteImage.astype(np.float64) / 255.0)
    paletteImage = paletteImage / np.max(paletteImage)
    paletteImage *= 255.0
    paletteImage = np.clip(paletteImage, 0, 255).astype(np.uint8)

    return paletteImage
