import os
import numpy as np
import logging
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image preprocessing function
def prepare_image(file_path):
    img = load_img(file_path, color_mode='rgb', target_size=(299, 299))  # Force RGB
    img_array = img_to_array(img)  # Shape: (299, 299, 3)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 299, 299, 3)
    return img_array

# Load the trained model
model = load_model('covid_classification_model.h5')
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure 'uploads' directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Class labels
full_index = ['COVID', 'Lung_Capacity', 'Normal', 'Viral_Pneumonia', 
              'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
# Top-4 relevant classes
index = ['COVID', 'Lung_Capacity', 'Normal', 'Viral_Pneumonia']

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/vaccination')
def vaccination():
    return render_template('vaccination.html')

@app.route('/test', methods=["GET", "POST"])
def test():
    prediction = None
    try:
        if request.method == "POST":
            if 'image' not in request.files:
                logging.error("No file part in the request")
                return "No file part", 400

            f = request.files['image']
            if f.filename == '':
                logging.error("No file selected")
                return "No file selected", 400

            # Save file
            filepath = os.path.join(uploads_dir, f.filename)
            f.save(filepath)
            logging.info(f"File saved to {filepath}")

            # Preprocess image
            img = prepare_image(filepath)
            logging.info(f"Image processed with shape: {img.shape}")

            # Predict
            predictions = model.predict(img)
            logging.info(f"Model raw prediction: {predictions[0]}")

            # Get top-4 predictions
            top_predictions = predictions[0].argsort()[-4:][::-1]
            top_classes = [full_index[i] for i in top_predictions]
            logging.info(f"Top 4 predicted classes: {top_classes}")
            top_probabilities = predictions[0][top_predictions]
            logging.info(f"Top 4 probabilities: {top_probabilities}")

            # Pick the best relevant class
            for predicted_class in top_classes:
                if predicted_class in index:
                    prediction = predicted_class
                    break

            if not prediction:
                prediction = "No relevant prediction found in top classes"

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return f"An error occurred: {e}", 500

    return render_template('test.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
