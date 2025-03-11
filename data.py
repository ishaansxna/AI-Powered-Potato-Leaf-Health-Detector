import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

# Define dataset paths
train_dir = r"C:\Users\Ishaan Saxena\Desktop\PlantVillage"
early_blight_dir = os.path.join(train_dir, "Potato___Early_blight")
late_blight_dir = os.path.join(train_dir, "Potato___Late_blight")
healthy_dir = os.path.join(train_dir, "Potato___healthy")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Dataset path does not exist: {train_dir}")

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    print(f"Checking folder: {folder}")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)
    return images, labels

# Load all images and labels
early_blight_images, early_blight_labels = load_images_from_folder(early_blight_dir, 0)
late_blight_images, late_blight_labels = load_images_from_folder(late_blight_dir, 1)
healthy_images, healthy_labels = load_images_from_folder(healthy_dir, 2)

# Combine all data
X = np.array(early_blight_images + late_blight_images + healthy_images)
y = np.array(early_blight_labels + late_blight_labels + healthy_labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert labels to categorical
num_classes = 3
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

# ResNet50 Model with Fine-Tuning
def create_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        BatchNormalization(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

resnet50_model = create_resnet50()

# CNN Model
def create_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn()

# Callbacks for Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train ResNet50 model
history_resnet50 = resnet50_model.fit(
    train_generator,
    validation_data=(X_test, y_test),
    epochs=5,
    callbacks=[early_stopping, reduce_lr]
)

# Train CNN model
history_cnn = cnn_model.fit(
    train_generator,
    validation_data=(X_test, y_test),
    epochs=5,
    callbacks=[early_stopping, reduce_lr]
)

# Save models
resnet50_model.save('resnet50_plant_disease_model.h5')
cnn_model.save('cnn_plant_disease_model.h5')

# Function to predict plant health using a model
def predict_image(image_path, model_path, class_labels):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(model_path)
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    print(f"Predicted class: {predicted_class}")
    return predicted_class

# Example usage
class_labels = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}
test_image_path = os.path.join(train_dir, "test.jpeg")

# Predict using ResNet50
print("ResNet50 Prediction:")
predict_image(test_image_path, 'resnet50_plant_disease_model.h5', class_labels)

# Predict using CNN
print("CNN Prediction:")
predict_image(test_image_path, 'cnn_plant_disease_model.h5', class_labels)

print("Models Trained and Saved Successfully!")
