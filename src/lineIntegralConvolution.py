import cv2
import numpy as np
import vectorField

def noiseImage(image, overallTone=0.7):
    '''
    Input:
    image - RGB image.
    overallTone - coefficient between 0 and 1 to determine strength of noise.

    Output:
    noiseImage - Grayscale image.

    Produces a white noise that matches the tone of the grayscale image. The probability that a pixel is white is proportional to the gray value.
    '''

    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    maxIntensity = np.max(grayImage)
    noiseImage = np.zeros(grayImage.shape).astype(np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            threshold = overallTone * (1 - grayImage[i, j]/maxIntensity)
            probability = np.random.uniform()
            if probability >= threshold:
                noiseImage[i, j] = 255
            else:
                noiseImage[i, j] = 0
    
    return noiseImage

def licSketch(image, steps):
    '''
    Input:
    image - RGB image.
    steps - number of steps to walk in LIC.
    
    Output:
    sketchImage - Grayscale image.

    Performs Line Integral Convolution on a white noise image and vector field. For every pixel, the algorithm walks, in the direction of the gradient at the pixel, for steps number of units and averages their intensities.
    '''

    vectors = vectorField.regionVectorField(image)
    gradients = np.arctan2(vectors[:, :, 1], vectors[:, :, 0])
    gradients = np.rad2deg(gradients)
    noise = noiseImage(image)

    def pixelDirection(angle):
        angle += 180
        x, y = 1, 0
        if angle > 0 and angle <= 22.5:
            x, y = 1, 0
        elif angle > 22.5 and angle <= 67.5:
            x, y = 1, -1
        elif angle > 67.5 and angle <= 112.5:
            x, y = 0, -1
        elif angle > 112.5 and angle <= 157.5:
            x, y = -1, -1
        elif angle > 157.5 and angle <= 202.5:
            x, y = -1, 0
        elif angle > 202.5 and angle <= 247.5:
            x, y = -1, 1
        elif angle > 247.5 and angle <= 292.5:
            x, y = 0, 1
        elif angle > 292.5 and angle <= 337.5:
            x, y = 1, 1
        elif angle > 337.5 and angle <= 360:
            x, y = 1, 0
        
        return x, y

    sketchImage = np.zeros(noise.shape)
    kernel = np.arange(steps)[::-1]
    for i in range(sketchImage.shape[0]):
        for j in range(sketchImage.shape[1]):
            count = 1
            smear = 0
            x, y = pixelDirection(gradients[i, j])

            for step in range(1, steps):
                if i+x*step < 0 or j+y*step < 0 or i+x*step >= sketchImage.shape[0] or j+y*step >= sketchImage.shape[1]:
                    break
                else:
                    count += 1
                    smear += noise[i+x*step, j+y*step] * kernel[step]
                    x, y = pixelDirection(gradients[i+x*step, j+y*step])
            
            x, y = pixelDirection(gradients[i, j])
            for step in range(1, steps):
                if i-x*step < 0 or j-y*step < 0 or i-x*step >= sketchImage.shape[0] or j-y*step >= sketchImage.shape[1]:
                    break
                else:
                    count += 1
                    smear += noise[i-x*step, j-y*step] * kernel[step]
                    x, y = pixelDirection(gradients[i-x*step, j-y*step])
            sketchImage[i, j] = smear // count
    sketchImage /= np.max(sketchImage)

    return sketchImage
