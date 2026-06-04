from flask import Flask, render_template, request
import cv2
import joblib
import os

from skimage.feature import hog

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = joblib.load("gesture_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = cv2.imread(
        filepath,
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (64, 64))

    features = hog(
        img,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2)
    )

    prediction = model.predict(
        [features]
    )[0]

    gesture_names = {
        "01_palm":"Palm ✋",
        "02_l":"L Gesture 🤟",
        "03_fist":"Fist ✊",
        "04_fist_moved":"Moved Fist 👊",
        "05_thumb":"Thumbs Up 👍",
        "06_index":"Index ☝",
        "07_ok":"OK Sign 👌",
        "08_palm_moved":"Moving Palm ✋",
        "09_c":"C Gesture 🤏",
        "10_down":"Down Gesture 👇"
    }

    result = gesture_names.get(
        prediction,
        prediction
    )

    return render_template(
        "result.html",
        result=result,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)