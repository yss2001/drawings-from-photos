import sys
import cv2
import numpy as np
import lineIntegralConvolution
import regionSegmentation
import depthBlend
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Input image missing!")
    exit()

inputImage = cv2.imread("../data/inputs/" + sys.argv[1])
inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_RGB2GRAY)
paletteImage = regionSegmentation.colorPalette(inputImage)
paletteImage = paletteImage.astype(np.float64) / 255.0
lineSketch = lineIntegralConvolution.licSketch(inputImage, 15)

ratio = 0.5
nprImage = np.zeros(paletteImage.shape)
nprImage[:, :, 0] = 1 - (1 - (1 - ratio)*lineSketch) * (1 - ratio * paletteImage[:, :, 0])
nprImage[:, :, 1] = 1 - (1 - (1 - ratio)*lineSketch) * (1 - ratio * paletteImage[:, :, 1])
nprImage[:, :, 2] = 1 - (1 - (1 - ratio)*lineSketch) * (1 - ratio * paletteImage[:, :, 2])
blendedImage = depthBlend.blender(nprImage * 255.0, grayImage, 0.1)

blendedImage = cv2.cvtColor(blendedImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("../data/outputs/Final_" + sys.argv[1], blendedImage)