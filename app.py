from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import io
import cv2
import onedal as da

app = Flask(__name__)

# Load the pre-trained model
model = da.load_model('path/to/model')

# Define the class labels
classes = ['crop', 'weed']

# Define the image size
img_size = (224, 224)

# Define the pre-processing function for the image
def preprocess_image(image):
    # Resize the image to the desired size
    image = image.resize(img_size)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Expand the dimensions of the array to match the expected input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize the image data
    image_array = image_array / 255.0
    # Return the preprocessed image
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        # Read the image file as a PIL image
        image = Image.open(io.BytesIO(image_file.read()))
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        # Make a prediction using the pre-trained model
        prediction = model.predict(preprocessed_image)
        # Get the predicted class label
        predicted_class = classes[np.argmax(prediction)]
        # Render the result page with the predicted class label
        return render_template('result.html', predicted_class=predicted_class)
    else:
        # Render the upload page
        return render_template('upload.html')
