# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:24:29 2026

@author: unangsunarya
"""

 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. Load Iris Dataset
# ==========================================

#iris = load_iris()



Aris=np.array(pd.read_csv("dataset/Iris.csv"))

Xa=Aris[:,1:5].astype(np.float64)
Ya=Aris[:,5]

target_names=['setosa', 'versicolor', 'virginica'] #nama2 kelas 

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Ya)

y=encoder.transform(Ya)
X=Xa


#X = iris.data
#y = iris.target

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nIris Classes:")
print(target_names)

# ==========================================
# 2. Data Preprocessing
# ==========================================

# Normalize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert target into one-hot encoding
y = to_categorical(y, num_classes=3)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# 3. Build Deep Neural Network
# ==========================================

model = tf.keras.Sequential([

    # Input layer
    tf.keras.layers.Input(shape=(4,)),

    # Hidden Layer 1
    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    # Hidden Layer 2
    tf.keras.layers.Dense(
        32,
        activation='relu'
    ),

    # Hidden Layer 3
    tf.keras.layers.Dense(
        16,
        activation='relu'
    ),

    # Output Layer
    tf.keras.layers.Dense(
        3,
        activation='softmax'
    )
])

# ==========================================
# 4. Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display network architecture
model.summary()

# ==========================================
# 5. Train DNN Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# 6. Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:",loss)
print("Test Accuracy:", np.round(accuracy*100))

# ==========================================
# 7. Prediction
# ==========================================

y_pred = model.predict(X_test)

predicted_classes = np.argmax(y_pred, axis=1)
actual_classes = np.argmax(y_test, axis=1)

print("\nPrediction Results:")
for i in range(len(predicted_classes)):
    print(
        "Actual:", target_names[actual_classes[i]],
        "| Predicted:", target_names[predicted_classes[i]],
        "| Probability:", np.round(y_pred[i], 3)
    )

# ==========================================
# 8. Plot Training Accuracy
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title('DNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 9. Plot Training Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title('DNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 10. Confusion Matrix
# ==========================================

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

plt.show()


print(model.layers[0].get_weights()[0])
