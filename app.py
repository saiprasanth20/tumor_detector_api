from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = load_model('brain_tumor_detector.h5')

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)  # Save in the uploads directory
            file.save(file_path)
            
            # Process the image and make a prediction
            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            
            # Determine the result
            result = "Tumor" if prediction[0][0] > 0.5 else "Healthy"
            return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
