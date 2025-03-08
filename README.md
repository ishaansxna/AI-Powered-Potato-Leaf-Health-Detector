# 🌱 AI-Powered Crop Disease Detector 🚀

## 🌟 Introduction
Welcome to the **AI-Powered Crop Disease Detector**! 🧠🌿 This project utilizes **Convolutional Neural Networks (CNNs)** to analyze plant leaf images and detect diseases. It aims to assist farmers and agricultural experts in identifying crop diseases early and accurately. 🌾🔍

## ✨ Features
- 📸 **Image-Based Detection** – Upload an image of a leaf, and the model will predict the disease.
- 🏷️ **Multi-Class Classification** – Identifies various crop diseases.
- 📊 **Real-Time Predictions** – Provides instant results using a trained deep learning model.
- 📈 **High Accuracy** – Built using CNN architectures like **VGG16, ResNet50, or MobileNet**.
- 🖥️ **User-Friendly Interface** – (Optional) Deploy using **Flask/FastAPI & Streamlit**.

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍
- **Deep Learning Framework:** TensorFlow / PyTorch 🧠
- **Image Processing:** OpenCV 📷
- **Web Framework (Optional):** Flask / FastAPI 🌐
- **Dataset:** PlantVillage 📊

## 📥 Dataset
The dataset should be placed in the specified directory:
```
C:\Users\Ishaan Saxena\Desktop\PlantVillage
```
Make sure the dataset is structured in the following format:
```
PlantVillage/
    ├── Class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    ├── Class_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    ├── Class_3/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
```
Download the dataset from:
[🌍 PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)

## 📌 How to Use
1. Install dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib opencv-python flask fastapi streamlit
   ```
2. Run the app using Flask or Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Upload an image of a plant leaf. 🌿
4. Get instant results on whether the leaf is **healthy** or **infected**. ✅❌

## 🏗️ Model Architecture
The model consists of:
- ✅ Convolutional layers with ReLU activation
- ✅ MaxPooling layers to reduce spatial dimensions
- ✅ Fully connected (Dense) layer with 512 neurons
- ✅ A final output layer with Softmax activation for multi-class classification

The CNN is based on architectures such as:
- **VGG16**
- **ResNet50**
- **MobileNetV2**
- **EfficientNet**

## 🚀 Training Process
The training data undergoes augmentation using `ImageDataGenerator`:
- Rescaling pixel values
- Random rotations, shifts, shear, zoom, and flips
- 80% training and 20% validation split

The model is compiled using:
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
Training is done using:
```python
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size,
          epochs=10)
```

## 💾 Saving and Loading the Model
After training, the model is saved as:
```python
model.save('potato_disease_model.h5')
```
To load the trained model:
```python
from tensorflow.keras.models import load_model
model = load_model('potato_disease_model.h5')
```

## 📈 Results
After training, the model's performance can be evaluated using validation accuracy and loss metrics.

## 🚀 Future Improvements
- ✅ Enhance accuracy using **Transfer Learning**.
- ✅ Expand dataset to include more plant species. 🌍
- ✅ Deploy on **Cloud** for mobile accessibility. ☁️📱
- ✅ Create an **Android App** for real-time disease detection. 📲

## 🤝 Contributing
Contributions are always welcome! 🎉
1. Fork the repo 🍴
2. Create a new branch 🔀
3. Commit your changes ✅
4. Push & submit a PR 🚀

---
💡 **Made with ❤️ for Farmers & AgriTech Innovators!** 🚜

