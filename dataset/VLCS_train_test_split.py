import os
import random
from sklearn.model_selection import train_test_split

# Path to the dataset directories
dataset_path = {
    'VOC2007': './VLCS/VOC2007',
    'LabelMe': './VLCS/LabelMe',
    'Caltech101': './VLCS/Caltech101',
    'SUN09': './VLCS/SUN09'
}

# Define the categories (example for PascalVOC)
categories = ['bird', 'car', 'chair', 'dog', 'person']  # Update these categories based on your dataset
category_to_label = {category: i for i, category in enumerate(categories)}

# Function to get all image paths and their corresponding numerical labels
def get_image_paths_and_labels(dataset_path, categories, category_to_label):
    image_paths = []
    labels = []
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
        for image_name in os.listdir(category_path):
            if image_name.endswith('.jpg'):
                image_paths.append(os.path.join(category, image_name))
                labels.append(category_to_label[category])
    return image_paths, labels

# Function to create train and test split files
def create_train_test_split_files(image_paths, labels, test_size=0.2):
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42)
    return train_paths, test_paths, train_labels, test_labels


# Directory to store the split files
split_dir = 'VLCS/splits'
os.makedirs(split_dir, exist_ok=True)

# Process each dataset
for dataset_name, dataset_path in dataset_path.items():
    image_paths, labels = get_image_paths_and_labels(dataset_path, categories, category_to_label)
    train_paths, test_paths, train_labels, test_labels = create_train_test_split_files(image_paths, labels)
    
    # Write train and test split files for each dataset
    train_file_path = os.path.join(split_dir, f'{dataset_name.lower()}_train.txt')
    test_file_path = os.path.join(split_dir, f'{dataset_name.lower()}_test.txt')
    
    with open(train_file_path, 'w') as f:
        for path, label in zip(train_paths, train_labels):
            domain_and_image = os.path.join(dataset_name, path)
            f.write(f'{domain_and_image} {label}\n')

    with open(test_file_path, 'w') as f:
        for path, label in zip(test_paths, test_labels):
            domain_and_image = os.path.join(dataset_name, path)
            f.write(f'{domain_and_image} {label}\n')

print("Train and test split files created successfully.")