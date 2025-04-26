import os
import cv2
import numpy as np
import pickle
import torch
from kornia.feature import HyNet, extract_patches_from_pyramid
from sklearn.cluster import KMeans


def loadImagePathsAndLAFs(baseDir, prefix, type):
    # Paths for the pickle files created during sift feature calculations
    imagePathsFile = os.path.join(baseDir, f"{prefix}_{type}_image_paths.pkl")
    LAFsFile = os.path.join(baseDir, f"{prefix}_{type}_sift_keypoint_LAFs.pkl")

    # Initialize variables
    imagePaths = None
    lafs = None

    # Load from files
    with open(imagePathsFile, 'rb') as f:
        imagePaths = pickle.load(f)

    with open(LAFsFile, 'rb') as f:
        lafs = pickle.load(f)

    return imagePaths, lafs


def extractHyNetDescriptors(imagePaths, lafs, hynet):
    # Initialize empty list
    descriptors = []

    # For each image
    for i in range(len(imagePaths)):
        # Read image in grayscale
        image = cv2.imread(imagePaths[i], cv2.IMREAD_GRAYSCALE)

        # Convert to torch tensor and bring to kornia format (normalized float values)
        imageTensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32) / 255

        # Do the same for laf
        lafTensor = torch.from_numpy(lafs[i]).unsqueeze(0).to(dtype=torch.float32)

        # Get patches of the image
        imagePatches = extract_patches_from_pyramid(img=imageTensor, laf=lafTensor, PS=32, normalize_lafs_before_extraction=False)
        # Reshape to the hynet input format
        imagePatches = imagePatches.squeeze(0)

        # Get HyNet descriptors from patches
        imageDescriptors = hynet(imagePatches)

        # Convert to numpy array and add to list
        descriptors.append(imageDescriptors.detach().cpu().numpy())

        print(i)

    return descriptors


def performKMeans(trainedModels, K, descriptors, clusterLabels, type):
    if type == "train":
        # Stack all features from train images into a single input matrix
        X = np.vstack(descriptors)

        # Create and train the model
        model = KMeans(n_clusters=K, random_state=537, verbose=True)
        model.fit(X)

        # Add the computed cluster labels for the current K
        clusterLabels[K] = model.labels_

        # Add model to trained models
        trainedModels[K] = model

    else:
        # Stack features of test images
        y = np.vstack(descriptors)

        # Get trained model
        model = trainedModels[K]

        # Predict labels for test images, and add to dict for current K
        labels = model.predict(y)
        clusterLabels[K] = labels


def computeHistograms(K, descriptors, clusterLabels, histograms):
    # Get cluster labels of all the features (for the current K = number of bins in histogram)
    allLabels = clusterLabels[K]

    # Initialize an empty array to hold histograms of every image in the set
    allHistograms = []

    # Last index checked in labels, start with 0
    lastIndex = 0

    # For every image, get image's descriptor
    for descriptor in descriptors:
        # Get number of features in the HyNet descriptor of the image (=number of labels for the image)
        labelCount = descriptor.shape[0]
        # Index to check until for the image, in all labels
        nextIndex = lastIndex + labelCount
        # Get cluster labels for the current image
        labels = allLabels[lastIndex:nextIndex]
        # Initialize histogram for the image with current number of bins
        histogram = np.zeros(K)

        # For each cluster label
        for label in labels:
            # Increase the number of words/features in current bag/bin by 1
            histogram[label] += 1

        # Normalize histogram (make sum of elements = 1)
        histogram = histogram / labelCount
        # Add to list
        allHistograms.append(histogram)

        # Update last index
        lastIndex = nextIndex

    # Save all histograms for the current K
    histograms[K] = allHistograms


def exportToFiles(descriptors, histograms, prefix, type, dataDir):
    fileNameOfHyNetDescriptors = f"{prefix}_{type}_hynet_descriptors.pkl"
    filePathOfHyNetDescriptors = os.path.join(dataDir, fileNameOfHyNetDescriptors)
    with open(filePathOfHyNetDescriptors, 'wb') as f:
        pickle.dump(descriptors, f)

    for K, allHistograms in histograms.items():
        fileNameOfHistograms = f"{prefix}_{type}_hynet_histograms_{K}.pkl"
        filePathOfHistograms = os.path.join(dataDir, fileNameOfHistograms)
        with open(filePathOfHistograms, 'wb') as f:
            pickle.dump(allHistograms, f)


def main(prefix, types, dataDir, hynet, Ks, trainedModels):

    for type in types:
        # Get image paths and corresponding laf (local affine frames) info
        imagePaths, lafs = loadImagePathsAndLAFs(dataDir, prefix, type)

        # Get HyNet descriptors of images
        descriptors = extractHyNetDescriptors(imagePaths, lafs, hynet)

        clusterLabels = {}
        histograms = {}
        for K in Ks:
            # Get cluster labels of HyNet descriptors
            performKMeans(trainedModels, K, descriptors, clusterLabels, type)
            # Compute bow representation of images using cluster labels
            computeHistograms(K, descriptors, clusterLabels, histograms)

        # Export data to files
        exportToFiles(descriptors, histograms, prefix, type, dataDir)


if __name__ == "__main__":
    # Work with data produced during sift feature extraction
    # TurCoins dataset
    prefix = "turcoins"
    types = ["train", "test"]
    dataDir = "/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/ProducedData/100_64_sklearn"
    # HyNet descriptor
    hynet = HyNet(pretrained=True)
    # Number of cluster centers
    Ks = [500, 100, 50]
    trainedModels = {}

    main(prefix, types, dataDir, hynet, Ks, trainedModels)

    # Repeat for Caltech
    prefix = "caltech"
    trainedModels = {}
    main(prefix, types, dataDir, hynet, Ks, trainedModels)