import os
import pandas as pd
import shutil

# Path to the CSV file
csv_file = 'C:/Users/hp/Desktop/IISC/Datasets_overview/testall.csv'

# Root directory for the train set
train_dir = 'C:/Users/hp/Desktop/IISC/test'

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to move images to the corresponding class directories
def move_images(row, source_dir, dest_dir):
    image_file = row['Originalname']
    class_name = row['Part']
    source_path = os.path.join(source_dir, image_file)
    dest_path = os.path.join(dest_dir, class_name, image_file)
    if os.path.exists(source_path):
        shutil.move(source_path, dest_path)
    else:
        print(f"File {source_path} not found.")

# Assuming all images are currently in a single directory
source_dir = 'C:/Users/hp/Desktop/IISC/Datasets_overview/testsource'

# Move images to the train set
df.apply(lambda row: move_images(row, source_dir, train_dir), axis=1)

print("Dataset organized successfully.")
