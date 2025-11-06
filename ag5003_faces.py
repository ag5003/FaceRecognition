import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from glob import glob

"""
Face Recognition - from class notes
1. Detect Image: locate faces within raw images using Haar cascade classifiers
2. Normalize: convert detected face images to grayscale and resize to a standard size
3. Extract Facial Features: use PCA (eigenfaces) to extract the most significant facial features and reduce dimensionality
4. Recognize Facial Image: classify a given face by comparing its features to the training set, verifying or identifying the character


For this project, need to follow 4 steps:
1. Segmentation: segment Ariel's faces and save to new folder
2. Load all characters: all other character's images are already segmented - need to load everything
3. Face recognition: use PCA (eigenvalues) to recognize faces and compare character labels against predictions
4. End-to-end Execution: call all functions
"""

#1. Segmentation
def segmentation(image_folder, output_folder, cascade_path):
    """
    Detect faces in images inside image_folder and save to output_folder.
    Saves files using the folder name of output_folder as base, e.g. "ariel_sharon1.png" - following other characters' naming convention

    Args:
        image_folder (str): Path to raw images ("Faces/ariel_sharon_raw")
        output_folder (str): Path to save segmented faces ("Faces/ariel_sharon")
        cascade_path (str): Path to Haar cascade XML file
    """
    os.makedirs(output_folder, exist_ok=True) #this to make sure i have output folder created
    
    #parameters for face detection
    save_ext=".png" #saves images as .png to ensure consistency with other characters' images
    scaleFactor=1.01 #small scale factor for more accurate detection
    minNeighbors=10 #higher value reduces false positives
    minSize=(30, 30) #minimum size of detected face

    #Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)

    #get base name of folder for saving images
    base_name = os.path.basename(output_folder.rstrip("/\\"))
    saved_counter = 0 #counter to generate unique file names

    raw_files = [] #stores raw files
    #collect all images with any of the below file extensions
    for ext in ("*.jpg", "*.jpeg", "*.png"): 
        raw_files.extend(glob(os.path.join(image_folder, ext)))
    raw_files = sorted(raw_files)
    print(f"Found {len(raw_files)} raw files in {image_folder}")

    #looping over all raw images
    for raw_path in raw_files:
        img = cv2.imread(raw_path) #reads the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale

        #detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

        #checking if more than one face is detected, only save the first one detected (usually largest one)
        if len(faces) > 0:
            #x and y are the coordinates of the top-left corner of the detected face bounding box in the image
            #w is the width of the bounding box
            #h is the height of the bounding box
            (x, y, w, h) = faces[0]  
            saved_counter += 1
            #crop image and resize to standard 100x100 size  
            face_crop = gray[y:y+h, x:x+w] #uses coordinates to crop the face from grayscale image
            face_resized = cv2.resize(face_crop, (100, 100))
            #saves image with appropriate name
            save_name = f"{base_name}{saved_counter}{save_ext}"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, face_resized)
            print(f"saved face: {save_path}")
        else:
            print(f"no faces detected in {os.path.basename(raw_path)} (skipped)")

    print(f"face detection completed for folder: {image_folder}. Total saved: {saved_counter}")


#2. Load character faces
def load_faces(dataset_folders):
    """
    Load all images from given character folders, resize to standard size,
    and flatten into 1D arrays for PCA.

    Args:
        dataset_folders (list of str): List of folder paths for each character

    Returns:
        flattened_images (np.array): Array of flattened images
        labels (list): List of character labels corresponding to each image
    """
    image_size=(100, 100) #this is the standard size for pca
    flattened_images = [] #stores flattened image arrays
    labels = [] #stores labels for each image

    #loop through each character folders
    for folder in dataset_folders:
        print(f"Loading images from: {folder}")

        #collect image files
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(glob(os.path.join(folder, ext)))
        files = sorted(files)
        print(f"found {len(files)} images")

        #process each image in the character folder 
        for fpath in files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE) #loads grayscale
            img = cv2.resize(img, image_size) #resize to standard
            flattened_images.append(img.flatten()) #flatten to 1D
            labels.append(os.path.basename(folder))  #use folder name as label
    
    #convert list to np array for pca
    flattened_images = np.array(flattened_images)
    print("----")
    print(labels[:30])
    return flattened_images, labels

#Face Recognition: using PCA eigenvalues
def recognize_faces(flattened_images, labels, test_size=0.3, random_state=42):
    """
    Train PCA eigenfaces on training set and predict labels for test set.
    Uses nearest neighbor in PCA space.

    Args:
        flattened_images (np.array): Flattened images for all characters
        labels (list): Corresponding character labels
        test_size: 0.3 as stated in assignment instructions
        random_state: 42 as stated in assignment instructions
    """

    #encode string labels into integers because scikit learn requires int
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    class_names = le.classes_

    #train/test - stratified sampling to keep the same distribution of images per character
    X_train, X_test, y_train, y_test = train_test_split(
        flattened_images, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded if len(labels_encoded)>0 else None
    )

    print("Train distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{class_names[u]}: {c}")
    
    print("Test distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{class_names[u]}: {c}")

    #PCA components: cannot exceed min(n_samples, n_features)
    n_components = min(50, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)

    #fit PCA only on training data, then transform both train and test
    X_train_pca = pca.fit_transform(X_train) #only fit_transform on X_train
    X_test_pca = pca.transform(X_test)

    #nearest neighbor classification in PCA space
    y_pred = []
    #loop through each test image - each test image is compared as a whole vector (all components at once) to all training images in PCA space
    for x in X_test_pca:
        distances = np.linalg.norm(X_train_pca - x, axis=1) #computes the Euclidean distance from this test image to every training image in PCA space
        nearest_idx = np.argmin(distances) #gets the index for the nearest training image
        y_pred.append(y_train[nearest_idx]) #append the label of the nearest training image to predictions

    accuracy = accuracy_score(y_test, y_pred) #scikit learn accuracy - comparing prediction to ground truth
    print(f"Recognition accuracy: {accuracy:.4f}")
    return accuracy

#4. End-to-end Execution
if __name__ == "__main__":
    haar_cascade_path = "haarcascade_frontalface_default.xml"
    raw_ariel_folder = "Faces/ariel_sharon_raw"     
    segmented_ariel_folder = "Faces/ariel_sharon"   

    #Step 1: segment images for ariel and save in new folder
    segmentation(raw_ariel_folder, segmented_ariel_folder, haar_cascade_path)

    characters = [
        "Faces/ariel_sharon",
        "Faces/chris_evans",
        "Faces/chris_hemsworth",
        "Faces/mark_ruffalo",
        "Faces/robert_downey_jr",
        "Faces/scarlett_johansson"
    ]

    #Step 2: load faces for all characters
    flattened_images, labels = load_faces(characters)
    unique, counts = np.unique(labels, return_counts=True)
    print("Loaded label distribution:")
    for u, c in zip(unique, counts):
        print(f"{u}: {c} images")
    print(f"Total images loaded: {len(labels)}")

    # Step 3: run recognition for all characters and get accuracy
    recognize_faces(flattened_images, labels, test_size=0.3, random_state=42)
