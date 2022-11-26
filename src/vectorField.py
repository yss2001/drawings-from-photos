import cv2
import numpy as np

def mergePixels(edges, initialComponents, finalComponents):
    '''
    Input:
    edges - list of 3-tuples (weight, x, y) of all edges.
    initialComponents - initial number of nodes.
    finalComponents - final number of nodes.

    Output:
    parents - list of parent pixels of each pixel.
    '''

    parents, ranks, components = np.arange(initialComponents), np.zeros(initialComponents), initialComponents

    def getParent(pixel):
        if parents[pixel] == pixel:
            return pixel
        parents[pixel] = getParent(parents[pixel])
        return parents[pixel]
    
    for cost, x, y in edges:
        if components <= finalComponents:
            break

        x, y = getParent(x), getParent(y)
        if x == y:
            continue
        if ranks[x] > ranks[y]:
            x, y = y, x
        if ranks[x] == ranks[y]:
            ranks[y] += 1
        parents[x] = y
        components -= 1
    
    for pixel in range(initialComponents):
        parents[pixel] = getParent(pixel)
    
    return parents

def segmentRegions(image, numberOfRegions):
    '''
    Input:
    image - RGB image.
    numberOfRegions - Number of regions to segment the image into.

    Output:
    segments - 2D array of labels for each pixel.
    labels - list of counts of labels.

    Converts the RGB image into LAB space and finds the Euclidean distances of each pixel from its horizontal, vertical and diagonal neighbours. These are considered as edges in the graph and sorted. Union-Find is performed to merge them until the number of components remaining are equal to the number of regions.
    '''

    labImage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    yList, xList = np.meshgrid(np.arange(labImage.shape[1]), np.arange(labImage.shape[0]))
    edges = []
    indices = xList * labImage.shape[1] + yList

    horizontalCost = np.sqrt(np.sum(np.square(labImage[:, :-1, :] - labImage[:, 1:, :]), axis=2))
    horizontalEdges = list(zip(horizontalCost.ravel(), indices[:, :-1].ravel(), indices[:, 1:].ravel()))
    edges.extend(horizontalEdges)

    verticalCost = np.sqrt(np.sum(np.square(labImage[:-1, :, :] - labImage[1:, :, :]), axis=2))
    verticalEdges = list(zip(verticalCost.ravel(), indices[:-1, :].ravel(), indices[1:, :].ravel()))
    edges.extend(verticalEdges)

    diagonal1Cost = np.sqrt(np.sum(np.square(labImage[:-1, :-1, :] - labImage[1:, 1:, :]), axis=2))
    diagonal1Edges = list(zip(diagonal1Cost.ravel(), indices[:-1, :-1].ravel(), indices[1:, 1:].ravel()))
    edges.extend(diagonal1Edges)

    diagonal2Cost = np.sqrt(np.sum(np.square(labImage[1:, :-1, :] - labImage[:-1, 1:, :]), axis=2))
    diagonal2Edges = list(zip(diagonal2Cost.ravel(), indices[1:, :-1].ravel(), indices[:-1, 1:].ravel()))
    edges.extend(diagonal2Edges)

    edges.sort()
    parents = mergePixels(edges, image.shape[0] * image.shape[1], numberOfRegions)

    labelDict = dict()
    segments = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    labels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            parent = parents[i * image.shape[1] + j]
            if parent not in labelDict:
                labelDict[parent] = len(labelDict)
                labels.append(0)
            segments[i, j] = labelDict[parent]
            labels[segments[i, j]] += 1

    return segments, labels

def regionVectorField(image, blurSigma=2, blurSize=7, varianceThreshold=0.5):
    '''
    Input:
    image - RGB image.
    blurSigma - sigma value for Gaussian blur.
    blurSize - kernel size for Gaussian blur.
    varianceThreshold - threshold for snapping mean vectors.

    Output:
    vectors - 3D array of vector directions at each pixel.
    '''

    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (blurSize, blurSize), blurSigma)
    gradientX = cv2.Sobel(blurImage, cv2.CV_64F, 1, 0, ksize=5)
    gradientY = cv2.Sobel(blurImage, cv2.CV_64F, 0, 1, ksize=5)
    vectors = np.dstack([gradientX, -gradientY]) / 255.0
    vectors = vectors * (1 - 2 * (vectors[:, :, 1] < 0))[:, :, np.newaxis]

    segments, labels = segmentRegions(image, image.shape[0] * image.shape[1] // 8)

    means, variances = np.zeros((len(labels), 2)), np.zeros(len(labels))
    xList, yList = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    means[segments[xList, yList]] += vectors[xList, yList, :]
    means /= np.array(labels)[:, np.newaxis]
    variances[segments[xList, yList]] += np.sum(np.square(vectors[xList, yList, :] - means[segments[xList, yList]]))
    variances /= np.array(labels)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if variances[segments[i, j]] > varianceThreshold:
                vectors[i, j, :] = means[segments[i, j]] 
    vectors /= (np.sum(vectors ** 2, axis=2) + 1e-2)[:, :, np.newaxis]

    return vectors   