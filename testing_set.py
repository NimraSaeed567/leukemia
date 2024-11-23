import os
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Path to the testing images folder
testing_images_folder = r"D:\Users\HP\Downloads\archive\C-NMC_Leukemia\testing_data\C-NMC_test_final_phase_data"

# List all image files in the folder
testing_image_files = [f for f in os.listdir(testing_images_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not testing_image_files:
    print("No image files found in the folder. Check the folder path and image formats.")
else:
    print(f"Number of image files found: {len(testing_image_files)}")

# Initialize a list to hold the features
testing_features = []

# Loop through the testing image files and extract features
for testing_image_file in testing_image_files:
    img_path = os.path.join(testing_images_folder, testing_image_file)
    
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize images
        img_array = image.img_to_array(img)  # Convert to array
        
        # Flatten the image array into a 1D vector
        flattened_signal = img_array.flatten()
        
        # Compute FFT for frequency domain analysis (optional)
        fft_signal = np.fft.fft(flattened_signal)
        freq_signal = np.abs(fft_signal)  # Magnitude of FFT
        
        # Append the FFT signal to the list of features
        testing_features.append(freq_signal)
    except Exception as e:
        print(f"Error processing {testing_image_file}: {e}")
        continue

# Ensure features were extracted
if len(testing_features) == 0:
    print("No features extracted. Check image processing code.")
else:
    print(f"Number of features extracted: {len(testing_features)}")

# Convert the list of testing features to a numpy array
testing_features = np.array(testing_features)

# Create a DataFrame with the features and filenames
if testing_features.size > 0:
    testing_features_df = pd.DataFrame(testing_features)
    testing_features_df['image_name'] = testing_image_files  # Add the image filenames to the DataFrame
    
    # Save the testing features to a CSV file
    testing_features_csv = 'testing_image_features.csv'
    testing_features_df.to_csv(testing_features_csv, index=False)
    print(f"Testing features saved to '{testing_features_csv}'")
else:
    print("Testing features array is empty. No CSV file saved.")
