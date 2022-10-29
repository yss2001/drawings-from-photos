import cv2
import numpy as np
from sklearn.cluster import KMeans

def colorPalette(image, numberOfColors=10):
    '''
    Input:
    image - RGB image.
    numberOfColors - Number of clusters.

    Output:
    paletteImage - RGB image.

    Converts the RGB image into LAB space and performs KMeans clustering on the colors to classify them into the specified number of clusters. Each original color is then reassigned to the closest centroid color and is converted back to RGB space.
    '''

    labImage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    labImage = labImage.reshape(labImage.shape[0] * labImage.shape[1], labImage.shape[2])

    model = KMeans(n_clusters=numberOfColors)
    labels = model.fit_predict(labImage)
    centers = model.cluster_centers_.round(0).astype(int)
    
    paletteImage = np.reshape(centers[labels], image.shape).astype(np.uint8)
    paletteImage = cv2.cvtColor(paletteImage, cv2.COLOR_Lab2RGB)

    return paletteImage