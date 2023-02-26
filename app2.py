from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
# Replace this with the path to your own pre-trained model file
model_file = os.path.join(os.getcwd(), 'weed_detection_model.pth')

# Define the classes for the model
classes = ['crop', 'weed']

# Define the OneAPI Image Processing Library (IPL) functions
def read_image(image_file):
    return np.array(Image.open(image_file))

def resize_image(image, size):
    return cv2.resize(image, size)

def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def detect_weed(image):
    # Load the pre-trained model
    model = load_model(model_file)

    # Preprocess the image
    image = resize_image(image, (224, 224))
    image = normalize_image(image)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # Run the model on the image
    outputs = model(torch.from_numpy(image).float())
    _, predicted = torch.max(outputs.data, 1)

    return classes[predicted.item()]

# Define the Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded image from the form
    image_file = request.files['image']

    # Read the image using PIL
    image = read_image(image_file)

    # Detect the presence of weed in the image
    result = detect_weed(image)

    # Return the result to the user
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
