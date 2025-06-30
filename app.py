import os
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classes as per training label order
classes = ['Fresh_Apple', 'Fresh_Banana', 'Fresh_Orange', 'Rotten_Apple', 'Rotten_Banana', 'Rotten_Orange']

# Load model once
model = load_model(os.path.join(APP_ROOT, 'model.h5'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename == "":
        return "No file selected"

    filename = uploaded_file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(save_path)

    # Prepare image
    img = image.load_img(save_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction_probs = model.predict(img_array)
    predicted_index = np.argmax(prediction_probs)
    print('Prediction probabilities:', prediction_probs)
    print('Predicted index:', predicted_index)
    prediction = classes[predicted_index]

    return render_template("template.html", image_name=filename, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
