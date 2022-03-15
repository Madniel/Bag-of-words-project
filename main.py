import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# TODO: versions of libraries that will be used:
#  Python 3.9 (you can use previous versions as well)
#  numpy 1.19.4
#  scikit-learn 0.22.2.post1
#  opencv-python 4.2.0.34
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(dataset_dir_path.iterdir()):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    x = []
    dim = (1920,1080) # 0.84375
    for img in y:
        image = cv2.resize(img,dim)
        x.append(image)
    return x


def project():
    np.random.seed(42)

    # TODO: fill the following values
    first_name = 'Michał'
    last_name = 'Daniel'

    data_path = Path('train')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)

    # TODO: create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.AKAZE_create()

    # TODO: train a vocabulary model and save it using pickle.dump function

    train_images, test_images, train_labels, test_labels = train_test_split(x, y, train_size=0.6,
                                                                            random_state=42, stratify=y)

    test_images, valid_images, test_labels, valid_labels = train_test_split(test_images, test_labels, train_size=0.5,
                                                                            random_state=42, stratify=test_labels)

    train_descriptors = [descriptor for image in train_images
                         for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]]
    print("Descriptors:", len(train_descriptors))

    NB_WORDS = 128

    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
    kmeans = kmeans.fit(train_descriptors)

    with open('vocab_model.p', 'wb') as vocab: pickle.dump(kmeans, vocab)
    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
    y_train = train_labels

    X_valid = apply_feature_transform(valid_images, feature_detector_descriptor, kmeans)
    y_valid = valid_labels

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, kmeans)
    y_test = test_labels

    # TODO: train a classifier and save it using pickle.dump function
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train ,y_train)

    with open('clf.p', 'wb') as solution:
        pickle.dump(classifier, solution)
    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    print(classifier.score(X_train, y_train))
    print(classifier.score(X_valid, y_valid))
    score = clf.score(X_test, y_test)
    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:  # Don't change the path here
        json.dump({'score': score}, score_file)


if __name__ == '__main__':
    project()