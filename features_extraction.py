import numpy as np
import pandas as pd
import os
import cv2

# Function to convert images to electrical signals with feature extraction
def images_to_signal_features(images, labels, scale_range=(0, 5), feature_count=100):
    feature_data = []  # To store the features for each image
    feature_labels = []  # To store corresponding labels

    min_val, max_val = scale_range  # Voltage range
    
    for image, label in zip(images, labels):
        try:
            # Flatten the image to a 1D signal
            flattened_image = image.flatten()

            # Normalize pixel values (0-255) to the scale range (e.g., 0-5)
            normalized_signal = (flattened_image - np.min(flattened_image)) / np.ptp(flattened_image)  # Ensure normalization is correct
            scaled_signal = normalized_signal * (max_val - min_val) + min_val  # Scale to the desired range

            # Feature extraction (downsampling to fixed size)
            step = max(1, len(scaled_signal) // feature_count)  # Ensure step size is large enough
            features = scaled_signal[::step][:feature_count]  # Extract 'feature_count' samples

            # Handle cases where extracted features are less than required
            if len(features) < feature_count:
                features = np.pad(features, (0, feature_count - len(features)), mode='constant')

            feature_data.append(features)
            feature_labels.append(label)
        except Exception as e:
            print(f"Error processing an image: {e}")
            continue  # Skip problematic images

    return np.array(feature_data, dtype='float32'), np.array(feature_labels)

# Function to load images with batch processing
def load_and_process_images_in_batches(base_dir, feature_count=100, batch_size=100):
    processed_data = []
    processed_labels = []

    # Iterate through all directories and process images in batches
    for root, _, files in os.walk(base_dir):
        images = []
        labels = []

        for file_name in files:
            if file_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                label = os.path.basename(root)  # Use folder name as the label
                file_path = os.path.join(root, file_name)

                try:
                    # Load the image in grayscale to reduce dimensionality
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    # Ensure the image is not None and has data
                    if image is None:
                        print(f"Skipping empty or corrupted image: {file_path}")
                        continue

                    # Resizing to a standard size (optional)
                    image = cv2.resize(image, (128, 128))  # Resize to a consistent shape, change as needed
                    
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
                    continue

                # Process images in batches
                if len(images) >= batch_size:
                    batch_features, batch_labels = images_to_signal_features(images, labels, feature_count=feature_count)
                    processed_data.append(batch_features)
                    processed_labels.append(batch_labels)
                    images, labels = [], []  # Reset for the next batch

        # Process any remaining images
        if images:
            batch_features, batch_labels = images_to_signal_features(images, labels, feature_count=feature_count)
            processed_data.append(batch_features)
            processed_labels.append(batch_labels)

    return np.vstack(processed_data), np.concatenate(processed_labels)

# Base directory
base_dir = r"D:\Users\hp\Downloads\archive\C-NMC_Leukemia"

# Process dataset
feature_count = 100  # Number of features per signal
batch_size = 100  # Process 100 images at a time
features, feature_labels = load_and_process_images_in_batches(base_dir, feature_count, batch_size)

# Save to a structured dataset (e.g., CSV file)
dataset = pd.DataFrame(features)
dataset['label'] = feature_labels
output_path = "leukemia_image_signals_dataset.csv"
dataset.to_csv(output_path, index=False)

# Print summary
print(f"Dataset saved to {output_path}")
print(f"Total samples: {len(features)}, Features per sample: {features.shape[1]}")

import pandas as pd

# Function to organize features into a structured dataset
def create_dataset_table(features, labels):
    # Convert the feature data into a DataFrame
    feature_columns = [f"Feature_{i+1}" for i in range(features.shape[1])]  # Dynamically generate column names
    df = pd.DataFrame(features, columns=feature_columns)
    
    # Add the labels column
    df['Label'] = labels
    
    return df

# Assuming 'features' and 'feature_labels' have been extracted from your image processing code
# Create the dataset table
dataset_df = create_dataset_table(features, feature_labels)

# Save the dataset to a CSV file
output_file = "leukemia_image_signals_dataset.csv"
dataset_df.to_csv(output_file, index=False)

# Print dataset summary
print(f"Dataset saved as {output_file}")
print(f"Total samples: {len(features)}, Features per sample: {features.shape[1]}")
print(dataset_df.head())  # Display the first few rows of the dataset for preview

