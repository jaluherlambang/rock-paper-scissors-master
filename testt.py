import cv2
import numpy as np
from keras.models import load_model

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

# Load the trained model
model = load_model("rock-paper-scissors-model-2.h5")

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    return image

# Load and preprocess the image you want to make predictions on
image_path = "C:/Users/jaluh/OneDrive/Dokumen/Dicoding_python/sample-rock.jpg"  # Replace with the actual path to your image
image = cv2.imread(image_path)
preprocessed_image = preprocess_image(image)

# Reshape the preprocessed image to match the input shape of the model
input_image = np.reshape(preprocessed_image, (1, 150, 150, 3))

# Make predictions
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions[0])
predicted_class_name = mapper(predicted_class)

print("Predicted class:", predicted_class_name)
print(predictions)