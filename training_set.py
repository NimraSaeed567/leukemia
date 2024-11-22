import os
import cv2
import matplotlib.pyplot as plt

# Base directory path
base_dir = r"D:\Users\HP\Downloads\archive\C-NMC_Leukemia\training_data"

# Function to load images from folders and subfolders
def load_images_with_subfolders(base_dir):
    images = []  # To store image data
    labels = []  # To store labels ('all' or 'hem')

    # Loop through the main folders (folder1, folder2, folder3)
    for main_folder in os.listdir(base_dir):
        main_folder_path = os.path.join(base_dir, main_folder)
        if os.path.isdir(main_folder_path):  # Check if it's a folder
            # Loop through the subdirectories ('all' and 'hem')
            for sub_folder in ['all', 'hem']:
                sub_folder_path = os.path.join(main_folder_path, sub_folder)
                if os.path.isdir(sub_folder_path):  # Ensure it exists
                    # Load all images in the subdirectory
                    for file_name in os.listdir(sub_folder_path):
                        if file_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # Added .bmp to the condition
                            file_path = os.path.join(sub_folder_path, file_name)
                            image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load image in color
                            images.append(image)
                            labels.append(sub_folder)  # Label as 'all' or 'hem'

    return images, labels

# Load the dataset
images, labels = load_images_with_subfolders(base_dir)

# Print dataset info
print(f"Total images loaded: {len(images)}")
print(f"Labels distribution: {dict((label, labels.count(label)) for label in set(labels))}")

# Visualize a sample image
if images:
    plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title(f"Label: {labels[0]}")
    plt.axis('off')
    plt.show()
else:
    print("No images found. Check the directory paths.")

