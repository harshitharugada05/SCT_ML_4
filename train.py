import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = r"dataset\hand gesture\leapGestRecog"

X = []
y = []

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        label = gesture

        for img_name in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (64, 64))

                features = hog(
                    img,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2)
                )

                X.append(features)
                y.append(label)

            except:
                pass

print("Images Loaded:", len(X))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "gesture_model.pkl")

print("Model saved successfully!")