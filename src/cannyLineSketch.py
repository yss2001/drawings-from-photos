import cv2
import numpy as np

def nonMaxSuppression(gradMagnitude, gradAngle):
    '''
    Input:
    gradMagnitude - [float16] Magnitude of the grayscale image's gradient.
    gradAngle - [float16] Angle (in radians) of the grayscale image's gradient.

    Output:
    thinnedEdgeImage - [int32] Grayscale image.

    Applies lower bound thresholding to perform edge thinning and remove unwanted points on edges.
    '''

    M, N = gradMagnitude.shape
    thinnedEdgeImage = np.zeros((M, N), dtype=np.int32)
    gradAngle = (gradAngle * 180) / np.pi
    gradAngle[gradAngle < 0] += 180

    for i in range(1, M):
        for j in range(1, N):
            try:
                local1 = 255
                local2 = 255
                if (0 <= gradAngle[i, j] < 22.5) or (157.5 <= gradAngle[i, j] <= 180):
                    local1 = gradMagnitude[i, j+1]
                    local2 = gradMagnitude[i, j-1]
                elif (22.5 <= gradAngle[i, j] < 67.5):
                    local1 = gradMagnitude[i+1, j-1]
                    local2 = gradMagnitude[i-1, j+1]
                elif (67.5 <= gradAngle[i, j] < 112.5):
                    local1 = gradMagnitude[i+1, j]
                    local2 = gradMagnitude[i-1, j]
                elif (112.5 <= gradAngle[i, j] <= 157.5):
                    local1 = gradMagnitude[i-1, j-1]
                    local2 = gradMagnitude[i+1, j+1]
                
                if (gradMagnitude[i, j] >= local1) and (gradMagnitude[i, j] >= local2):
                    thinnedEdgeImage[i, j] = gradMagnitude[i, j]
                
            except IndexError as e:
                pass
    
    return thinnedEdgeImage

def doubleThresholding(thinnedEdgeImage, lowThresholdRatio, highThresholdRatio):
    '''
    Input:
    thinnedEdgeImage - [int32] Grayscale image.
    lowThresholdRatio - Floating point number from 0 to 1.
    highThresholdRatio - Floating point number from 0 to 1.

    Output:
    thresholdedEdgeImage - [uint8] Grayscale image.
    strongValue - Integer number.
    weakValue - Integer number.

    Marks all the pixels with intensities greater than the higher threshold as strong, and all the pixels with intensities lying between the lower and higher threshold as weak.
    '''

    M, N = thinnedEdgeImage.shape
    thresholdedEdgeImage = np.zeros((M, N)).astype(np.uint8)
    highThreshold = np.max(thinnedEdgeImage) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    strongValue = 255
    weakValue = 1

    mask1 = (thinnedEdgeImage >= highThreshold)
    mask2 = (thinnedEdgeImage < highThreshold) & (thinnedEdgeImage >= lowThreshold)
    thresholdedEdgeImage[mask1] = strongValue
    thresholdedEdgeImage[mask2] = weakValue

    return thresholdedEdgeImage, strongValue, weakValue


def hysteresis(thresholdedEdgeImage, strongValue, weakValue):
    '''
    Input:
    thresholdedEdgeImage - [uint8] Grayscale image.
    strongValue - Integer number.
    weakValue - Integer number.

    Output:
    cannyEdgeImage - [uint8] Binary image.

    Using eight-connectivity, each weak pixel is checked if it is adjacent to any strong pixel. If so, then such a pixel is converted into a strong pixel. If not, it is suppressed into the background.
    '''

    M, N = thresholdedEdgeImage.shape
    cannyEdgeImage = np.copy(thresholdedEdgeImage)

    for i in range(1, M):
        for j in range(1, N):
            if (cannyEdgeImage[i, j] == weakValue):
                try:
                    if (np.any(cannyEdgeImage[i-1 : i+2, j-1 : j+2] == strongValue)):
                        cannyEdgeImage[i, j] = strongValue
                    else:
                        cannyEdgeImage[i, j] = 0
                except IndexError as e:
                    pass
    
    return cannyEdgeImage



def cannyEdgeDetector(image, dilate=False, lineThickness=3):
    '''
    Input:
    image - [uint8] RGB image.
    dilate - Boolean.
    lineThickness - Integer value.

    Output:
    cannyEdgeImage - [uint8] Binary image.

    Applies the Canny Edge Detection algorithm on the input image to produce an edge image. Optionally performs dilation on the edge image to control the thickness of line.
    '''
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (3, 3), 1.5)

    sobelX = cv2.Sobel(blurredImage, cv2.CV_64F, 1, 0, ksize=3)
    sobelX = np.abs(sobelX).astype(np.uint8)
    sobelY = cv2.Sobel(blurredImage, cv2.CV_64F, 0, 1, ksize=3)
    sobelY = np.abs(sobelY).astype(np.uint8)

    gradMagnitude = np.hypot(sobelX, sobelY)
    gradAngle = np.arctan2(sobelY, sobelX)

    thinnedEdgeImage = nonMaxSuppression(gradMagnitude, gradAngle)
    thresholdedEdgeImage, strongValue, weakValue = doubleThresholding(thinnedEdgeImage, 0.1, 0.25)
    cannyEdgeImage = hysteresis(thresholdedEdgeImage, strongValue, weakValue).astype(np.uint8)

    if (dilate):
        structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lineThickness, lineThickness))
        cannyEdgeImage = cv2.dilate(cannyEdgeImage, structuringElement)
    
    return cannyEdgeImage
