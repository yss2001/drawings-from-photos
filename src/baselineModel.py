import sys
import cv2
import numpy as np
import kMeansColorPalette
import cannyLineSketch
import depthBlend

if len(sys.argv) < 2:
    print("Input image missing!")
    exit()

inputImage = cv2.imread("../data/inputs/" + sys.argv[1])
inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_RGB2GRAY)
paletteImage = kMeansColorPalette.colorPalette(inputImage, 7)
lineSketch = cannyLineSketch.cannyEdgeDetector(inputImage, False, 3)

mask = np.invert(lineSketch == 255)
edgeColor = np.array([32, 32, 32]).astype(np.uint8)
nprImage = np.zeros(inputImage.shape).astype(np.uint8)
nprImage += edgeColor
nprImage[mask, :] = paletteImage[mask, :]
blendedImage = depthBlend.blender(nprImage, grayImage, 0.5)

blendedImage = cv2.cvtColor(blendedImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("../data/outputs/Baseline_" + sys.argv[1], blendedImage)


