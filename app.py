# Import necessary libraries
from flask import Response
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Init Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and labels
model = load_model('keras_model.h5', compile=False)
with open('labels.txt', 'r') as f:
    class_names = f.readlines()

# Disable scientific notation
np.set_printoptions(suppress=True)

# Image preprocessing & prediction function


def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index].strip()[2:], float(prediction[0][index])

# Routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    label, confidence = classify_image(filepath)
    return render_template('result.html', label=label, confidence=confidence, image_path=filepath)


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess for model
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 255.0
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = img_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        # Add prediction text to frame
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# --- Route to serve video feed ---


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Route to live camera page ---


@app.route('/live')
def live():
    return render_template('live.html')


if __name__ == '__main__':
    app.run(debug=True)


