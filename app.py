from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define class labels
class_labels = {
    0: "Healthy",
    1: "Early Blight",
    2: "Late Blight"
}

# Load CNN (TensorFlow Model)
try:
    cnn_model = tf.keras.models.load_model("cnn_plant_disease_model.h5")
    print("✅ CNN Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading CNN Model: {e}")

# Load ResNet50 (TensorFlow Model)
try:
    resnet_model = tf.keras.models.load_model("resnet50_plant_disease_model.h5")
    print("✅ ResNet50 Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading ResNet50 Model: {e}")

# Image Preprocessing for TensorFlow Models
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    # Select model
    model_type = request.args.get("model", "cnn")  # Default to CNN

    # Use the correct model based on selection
    if model_type == "cnn":
        if "cnn_model" not in globals():
            return jsonify({"error": "CNN model not loaded"}), 500
        processed_image = preprocess_image(image)
        prediction = cnn_model.predict(processed_image)
    else:
        if "resnet_model" not in globals():
            return jsonify({"error": "ResNet50 model not loaded"}), 500
        processed_image = preprocess_image(image)
        prediction = resnet_model.predict(processed_image)

    predicted_class = np.argmax(prediction)  # Get class index
    disease_name = class_labels.get(predicted_class, "Unknown")  # Convert index to label

    return jsonify({"result": disease_name})  # Return meaningful label

if __name__ == "__main__":
    app.run(debug=True)
