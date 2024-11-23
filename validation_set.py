import os
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path to the images folder
images_folder = r"D:\Users\HP\Downloads\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data"

# List all image files in the folder (assuming they are in standard image formats like .jpg, .png)
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Initialize a list to hold image data
images_data = []

# Loop through the image files and load them
for image_file in image_files:
    img_path = os.path.join(images_folder, image_file)
    
    # Load image using Keras' image.load_img() method
    img = image.load_img(img_path, target_size=(224, 224))  # Resize images to 224x224, you can change the size
    img_array = image.img_to_array(img)  # Convert the image to an array
    
    # Append to the list
    images_data.append(img_array)

# Convert list to numpy array
images_data = np.array(images_data)

# Optionally, you can normalize the pixel values to [0, 1] range by dividing by 255
images_data = images_data / 255.0

# Optionally, display the first image to check
plt.imshow(images_data[0])  # Display the first image in the dataset
plt.show()

# Print the shape of the data (number of images, height, width, channels)
print(f"Shape of the images data: {images_data.shape}")

# Path to the images folder
images_folder = r"D:\Users\HP\Downloads\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data"

# List all image files in the folder (assuming they are in standard image formats like .jpg, .png)
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Initialize a list to hold the features
features = []

# Loop through the image files and extract features
for image_file in image_files:
    img_path = os.path.join(images_folder, image_file)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # Flatten the image array into a 1D vector (electrical signal form)
    flattened_signal = img_array.flatten()
    
    # Optionally, we can take the FFT of the signal to analyze it in the frequency domain
    fft_signal = np.fft.fft(flattened_signal)
    freq_signal = np.abs(fft_signal)  # Get the magnitude of the FFT (electrical signal in frequency domain)
    
    # Append the FFT signal to the list of features
    features.append(freq_signal)

# Convert the list of features to a numpy array
features = np.array(features)

# Optional: Create a DataFrame to hold the features and image filenames
features_df = pd.DataFrame(features)
features_df['image_name'] = image_files  # Add the image filenames to the DataFrame

# Save the features to a CSV file (optional)
features_df.to_csv('image_features_electrical_signals.csv', index=False)

# Print the shape of the features
print(f"Shape of the features: {features.shape}")

