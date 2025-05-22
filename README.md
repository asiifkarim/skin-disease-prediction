# 🧠 Skin Disease Classification using CNN

This project utilizes **deep learning** to classify images of various skin diseases using **Convolutional Neural Networks (CNNs)**. It is based on the Kaggle dataset **[Skin Diseases](https://www.kaggle.com/datasets/ascanipek/skin-diseases)** and includes preprocessing, model training, evaluation, and visualizations.

---

## 📁 Dataset

- **Source:** [Kaggle: Skin Diseases](https://www.kaggle.com/datasets/ascanipek/skin-diseases)
- **Categories:** Multiple classes of common skin conditions (e.g., acne, eczema, melanoma, etc.)
- **Format:** Image folders per class

The dataset was downloaded using `kagglehub` and organized into a balanced set by:
- Removing irrelevant or low-quality classes
- Balancing all classes to have the same number of images

---

## 📦 Project Structure

skin-disease-prediction/
│
├── data/ # Original and processed image datasets
├── Skin_Disease_classfication.ipynb # Main notebook for training and evaluation
├── model/ # Trained model and saved weights (if exported)
├── requirements.txt # Required Python packages
└── README.md # Project documentation

## Model Overview
Framework: TensorFlow / Keras

Architecture:

Multiple Convolutional Layers

ReLU Activation

MaxPooling Layers

Dropout for regularization

Dense layers

Final softmax layer for multi-class classification

```model = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```
## 📈 Results
Achieved 97% accuracy on validation data (replace with actual result)

Trained for 25+ epochs with early stopping and checkpointing

