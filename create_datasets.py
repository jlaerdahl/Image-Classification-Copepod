import os
import shutil
import random

def split_data(input_folder, output_folder, train_ratio, val_ratio):
    classes = os.listdir(input_folder)

    for class_name in classes:
        class_path = os.path.join(input_folder, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_size = int(len(images) * train_ratio)
        val_size = int(len(images) * val_ratio)

        train_images = images[:train_size]
        val_images = images[train_size:(train_size + val_size)]
        test_images = images[(train_size + val_size):]

        for image in train_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(output_folder, "train", class_name, image)
            shutil.copy(source_path, destination_path)

        for image in val_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(output_folder, "validate", class_name, image)
            shutil.copy(source_path, destination_path)

        for image in test_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(output_folder, "test", class_name, image)
            shutil.copy(source_path, destination_path)

# Example usage
input_folder = "data"
output_folder = "new_data"
train_ratio = 0.7
val_ratio = 0.15

# Create output directories
for dataset_type in ["train", "validate", "test"]:
    for class_name in ["copepod", "egg", "nauplii"]:
        os.makedirs(os.path.join(output_folder, dataset_type, class_name), exist_ok=True)

split_data(input_folder, output_folder, train_ratio, val_ratio)
