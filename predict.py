import cv2
import joblib
from skimage.feature import hog

model = joblib.load("gesture_model.pkl")

image_path = input("Enter image path: ")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (64, 64))

features = hog(
    img,
    pixels_per_cell=(8,8),
    cells_per_block=(2,2)
)

prediction = model.predict([features])

print("Predicted Gesture:", prediction[0])