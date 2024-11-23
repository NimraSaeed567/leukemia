#Features Extraction 
import pandas as pd

# Paths to the CSV files
image_features_csv = 'image_features_electrical_signals.csv'  # CSV containing image features extracted earlier
validation_labels_csv = r"D:\Users\HP\Downloads\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data_labels.csv"  # CSV with labels

# Import the image features CSV
image_features_df = pd.read_csv(image_features_csv)

# Import the validation labels CSV
validation_labels_df = pd.read_csv(validation_labels_csv)

# Rename 'new_names' to 'image_name' in validation_labels_df for consistency
validation_labels_df.rename(columns={'new_names': 'image_name'}, inplace=True)

# Merge the two DataFrames on the 'image_name' column
merged_df = pd.merge(image_features_df, validation_labels_df, on='image_name', how='inner')

# Display the merged DataFrame
print("\nMerged DataFrame (features + labels):")
print(merged_df.head())

# Optionally, save the merged data to a new CSV file
merged_df.to_csv('merged_image_data_with_labels.csv', index=False)
