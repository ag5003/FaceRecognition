Face Recognition Project - README

Overview
This project implements a face recognition system for a set of Avengers characters including Ariel Sharon. This is done by following four main steps:

1. Face detection / Segmentation: I use Haar cascade classifiers to detect faces in raw images. For this project, only Ariel Sharon’s raw images required segmentation; all other characters’ images were already pre-segmented. Detected faces are converted to grayscale, resized to a standard 100x100 pixels, and saved

2. Load dataset: all character images (Ariel Sharon plus the Avengers) are loaded into memory. Each image is flattened into a 1D vector of pixel values, and a corresponding label is stored

3. Feature extraction using PCA (Eigenfaces): PCA is applied to the flattened images to extract the principal components (eigenfaces), reducing dimensionality while capturing the most important facial features. Only the training set is used to fit the PCA transformation, and the test set is projected into this PCA space for evaluation

4. Face recognition: a nearest neighbor classifier in PCA space is used to predict the identity of each test image. For each test image, the Euclidean distance to all training images in PCA space is computed, and the label of the closest training image is assigned as the prediction. Accuracy is calculated with scikit-learn by comparing predicted labels to the ground truth test labels.

Logic of the Code

1. Segmentation (def segmentation)
* Loops over each raw image in the Ariel Sharon folder
* Converts to grayscale and uses Haar cascades to detect faces
* Saves the first detected face per image

2. Loading Faces (def load_faces)
* Loops through all character folders
* Loads all images, converts to grayscale, resizes to 100x100 pixels, flattens into vectors, and stores labels

3. Recognition (def recognize_faces)
* Encodes labels as integers (for scikit-learn)
* Splits data into training and test sets using stratified sampling (test_size=0.3, random_state=42) to maintain class distribution
* Performs PCA on training data to extract up to 50 principal components
* For each test image, computes Euclidean distance to all training images in PCA space and assigns the nearest neighbor label
* Computes accuracy by comparing predicted labels to true labels

4. Main Execution
* Segments Ariel Sharon’s raw images
* Loads all character images
* Runs PCA-based recognition and prints recognition accuracy and class distributions

Problems Found
1. File naming inconsistencies
* Ariel Sharon’s raw images were originally named differently (0.jpg vs. chris_evans1.png). This caused issues with loading and labeling. The solution was to rename and save images consistently

2. Mixed image formats
* Some folders contained .png while others had .jpg. I implemented the code to now accept multiple extensions

3. Extra faces detected / duplicates
* Initially, multiple faces or false positives were saved for Ariel Sharon, causing class imbalance.
* Fixed by saving only the first detected face per image and skipping images with no detection.

Potential Improvements

1. Tuning Haar cascade parameters
* Adjust scaleFactor, minNeighbors, and minSize to improve face detection quality and reduce false positives. Currently, it has been tuned to achieve 0.6333 accuracy (as per assignment instructions, needed to achieve > 60%)

2. Data augmentation
* Adding small rotations, flips, or brightness changes could make the classifier more robust. This could be done by applying OpenCV or torchvision image transformations to each training image before flattening and feeding it into PCA

3. Cross-validation
* Instead of a single train/test split, cross-validation could give a more reliable estimate of performance. This could be done if we had more data, since with 50 total images the test size would currently be too small. 

4. Better preprocessing
* Histogram equalization, alignment, or facial landmark normalization could improve PCA feature quality. This could be implemented using OpenCV functions like cv2.equalizeHist for contrast before resizing

