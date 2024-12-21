import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()


def load_images_from_folder(folder, filenames):
    images = []
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            if img is not None:
                img = ImageOps.fit(img, (224, 224))  # Resize image while maintaining aspect ratio
                new_img = Image.new("RGB", (224, 224), (255, 255, 255))
                new_img.paste(img, ((224 - img.size[0]) // 2, (224 - img.size[1]) // 2))
                new_img = np.array(new_img) / 255.0  # Normalize pixel values
                images.append(new_img)
    return np.array(images)

def load_labels_from_csv(file_path):
    df = pd.read_csv(file_path)

    training_data = df[df['is_training'] == 1]
    validation_data = df[df['is_training'] == 0]

    y_train, train_filenames = training_data['bmi'].values, training_data['name'].values
    y_val, val_filenames = validation_data['bmi'].values, validation_data['name'].values

    return y_train, train_filenames, y_val, val_filenames

def load_test_labels_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['bmi'].values, df['name'].values

def load_data(image_folder, dataset_csv, testset_csv):
    y_train, train_filenames, y_val, val_filenames = load_labels_from_csv(dataset_csv)
    x_train = load_images_from_folder(image_folder, train_filenames)
    x_val = load_images_from_folder(image_folder, val_filenames)

    y_test, test_filenames = load_test_labels_from_csv(testset_csv)
    x_test = load_images_from_folder(image_folder, test_filenames)

    print(f'Loaded {x_train.shape[0]} training images')
    print(f'Loaded {y_train.shape[0]} training labels')
    print(f'Loaded {x_val.shape[0]} validation images')
    print(f'Loaded {y_val.shape[0]} validation labels')
    print(f'Loaded {x_test.shape[0]} test images')
    print(f'Loaded {y_test.shape[0]} test labels')

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_test_data(image_folder, testset_csv):
    y_test, test_filenames = load_test_labels_from_csv(testset_csv)
    x_test = load_images_from_folder(image_folder, test_filenames)

    print(f'Loaded {x_test.shape[0]} test images')
    print(f'Loaded {y_test.shape[0]} test labels')

    return x_test, y_test, test_filenames
