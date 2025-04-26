import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans


def getImagePathsAndClassLabels(baseDir):
    # Initialize lists to hold image paths and class labels
    imagePaths = []
    classLabels = []

    # Passing through each sub-directory (class labels) in the base directory
    for classDir in os.listdir(baseDir):
        # Get path of class directory
        classPath = os.path.join(baseDir, classDir)

        # Ensure class path is a directory
        if os.path.isdir(classPath):

            # For each image file in the class directory
            for imageFile in os.listdir(classPath):

                # If the file is an image
                if imageFile.endswith('.jpg'):
                    # Add the image path to the list
                    imagePaths.append(os.path.join(classPath, imageFile))
                    # Add associated class label to the list
                    classLabels.append(classDir)

    return imagePaths, classLabels


def buildLafFromKeypoint(keypoint):
    # Create an empty LAF matrix of shape (2,3)
    laf = np.zeros((2, 3))

    # Get laf scale using keypoint size (in opencv docs "size" property of keypoint object is defined as
    # "diameter of the meaningful keypoint neighborhood", whereas in kornia, scale as interpreted as the radius
    # from keypoint center, so we divide by 2)
    scale = keypoint.size / 2
    # Get laf angle using keypoint angle (convert to radians from degrees)
    angle = np.deg2rad(keypoint.angle)
    # Get keypoint center coordinates
    x, y = keypoint.pt

    # Set elements of the LAF matrix
    laf[0, 0] = scale * np.cos(angle)
    laf[0, 1] = -scale * np.sin(angle)
    laf[1, 0] = scale * np.sin(angle)
    laf[1, 1] = scale * np.cos(angle)
    laf[0, 2] = x
    laf[1, 2] = y

    return laf


def performSIFT(imagePaths):
    # Initialize keypoint LAF (local affine frame) and descriptor lists
    keypointLAFs = []
    descriptors = []

    # Create sift feature extractor
    sift = cv2.SIFT.create(nfeatures=64)

    # For each image
    for imagePath in imagePaths:
        # Read image in grayscale
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        # Get keypoints and corresponding descriptors of the image
        k, d = sift.detectAndCompute(image=image, mask=None)

        # Extract keypoint LAFs from keypoint objects
        # Kornia accepts LAFs of shape (2,3) where the 2x2 square part (0:2, 0:2) holds information about
        # the scale and orientation, and last (third) column (0:2, 2) holds the LAF center (x, y coordinates of
        # the SIFT keypoint object)
        imageLAFs = None
        for keypoint in k:
            laf = buildLafFromKeypoint(keypoint)
            laf = np.expand_dims(laf, axis=0)
            imageLAFs = np.concatenate((imageLAFs, laf), axis=0) if imageLAFs is not None else laf

        # Append to lists
        keypointLAFs.append(imageLAFs)
        descriptors.append(d)

    return keypointLAFs, descriptors


def performKMeans(trainedModels, K, descriptors, clusterLabels, dirType):

    if dirType == 'train':
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
        # Get number of features in the SIFT descriptor of the image (=number of labels for the image)
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


def exportToFiles(imagePaths, classLabels, keypointLAFs, descriptors, histograms, prefix, outputDirPath, dirType):
    fileNameOfImagePaths = f"{prefix}_{dirType}_image_paths.pkl"
    filePathOfImagePaths = os.path.join(outputDirPath, fileNameOfImagePaths)
    with open(filePathOfImagePaths, 'wb') as f:
        pickle.dump(imagePaths, f)

    fileNameOfClassLabels = f"{prefix}_{dirType}_class_labels.pkl"
    filePathOfClassLabels = os.path.join(outputDirPath, fileNameOfClassLabels)
    with open(filePathOfClassLabels, 'wb') as f:
        pickle.dump(classLabels, f)

    fileNameOfSiftKeypointLAFs = f"{prefix}_{dirType}_sift_keypoint_LAFs.pkl"
    filePathOfSiftKeypointLAFs = os.path.join(outputDirPath, fileNameOfSiftKeypointLAFs)
    with open(filePathOfSiftKeypointLAFs, 'wb') as f:
        pickle.dump(keypointLAFs, f)

    fileNameOfSiftDescriptors = f"{prefix}_{dirType}_sift_descriptors.pkl"
    filePathOfSiftDescriptors = os.path.join(outputDirPath, fileNameOfSiftDescriptors)
    with open(filePathOfSiftDescriptors, 'wb') as f:
        pickle.dump(descriptors, f)

    for K, allHistograms in histograms.items():
        fileNameOfHistograms = f"{prefix}_{dirType}_sift_histograms_{K}.pkl"
        filePathOfHistograms = os.path.join(outputDirPath, fileNameOfHistograms)
        with open(filePathOfHistograms, 'wb') as f:
            pickle.dump(allHistograms, f)


def main(prefix, dirTypes, dirPaths, outputDirPath, Ks, trainedModels):

    for i in range(0, len(dirTypes)):
        # Get image paths and corresponding class labels
        imagePaths, classLabels = getImagePathsAndClassLabels(dirPaths[i])

        # Get LAFs (local affine frames) of SIFT keypoints and descriptors of images
        keypointLAFs, descriptors = performSIFT(imagePaths)

        clusterLabels = {}
        histograms = {}
        for K in Ks:
            # Get cluster labels of SIFT descriptors
            performKMeans(trainedModels, K, descriptors, clusterLabels, dirTypes[i])
            # Compute bow representation of images using cluster labels
            computeHistograms(K, descriptors, clusterLabels, histograms)

        # Export data to files
        exportToFiles(imagePaths, classLabels, keypointLAFs, descriptors, histograms, prefix, outputDirPath, dirTypes[i])


if __name__ == '__main__':
    # Get paths of Turcoin train and test directories
    prefix = "turcoins"
    dataSetName = "TurCoins"
    dirTypes = ["train", "test"]
    dirPaths = [f"/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/Data/{dataSetName}/{dirTypes[0]}",
                f"/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/Data/{dataSetName}/{dirTypes[1]}"]
    outputDirPath = "/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/ProducedData/100_64_sklearn"
    # Set different number of cluster centers for kmeans clustering
    Ks = [500, 100, 50]
    trainedModels = {}
    # Compute bag of features histogram from sift descriptors
    main(prefix, dirTypes, dirPaths, outputDirPath, Ks, trainedModels)


    # Repeat for Caltech dataset
    prefix = "caltech"
    dataSetName = "Caltech20"
    dirPaths = [f"/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/Data/{dataSetName}/{dirTypes[0]}",
                f"/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/Data/{dataSetName}/{dirTypes[1]}"]
    trainedModels = {}
    main(prefix, dirTypes, dirPaths, outputDirPath, Ks, trainedModels)
