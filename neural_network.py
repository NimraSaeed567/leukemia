import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = "D:/Users/leukemia/leukemia_image_signals_dataset.csv"
df = pd.read_csv(data_path)

# Check the first few rows of the dataset
print(df.head())

# Split the dataset into features (X) and target labels (y)
X = df.drop('Label', axis=1)  # Drop the label column
y = df['Label']  # The target variable (Label)

# Ensure that the labels are categorical
# Convert labels to numerical format if needed (assuming the labels are categorical strings)
y = pd.factorize(y)[0]

# Convert labels to one-hot encoding for multiclass classification
y = to_categorical(y, num_classes=4)  # Set the number of classes to 4 (as an example)

# Standardize the feature data (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the neural network model
model = Sequential()

# Add an input layer with the number of features
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Add a hidden layer
model.add(Dense(32, activation='relu'))

# Add an output layer with 4 units (one for each class) and softmax activation
model.add(Dense(4, activation='softmax'))  # 4 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Optionally, save the model
# model.save('leukemia_classification_model.h5')
