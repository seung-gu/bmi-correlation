import argparse
import json

import tensorflow as tf
from load_dataset import load_data
from models.resnet50v2 import ResNet50V2Model
from models.vgg16 import VGG16Model

# Ensure that the GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--model', type=str, required=True, help='Model name (RESNET50V2 or VGG16)')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
args = parser.parse_args()

# Load configuration of training
if args.model == 'RESNET50V2':
    with open('models/resnet50v2.json', 'r') as config_file:
        config = json.load(config_file)
    model = ResNet50V2Model(input_shape=(224, 224, 3), dropout=config['dropout'], dense=config['dense'])
elif args.model == 'VGG16':
    with open('models/vgg16.json', 'r') as config_file:
        config = json.load(config_file)
    model = VGG16Model(input_shape=(224, 224, 3), dropout=config['dropout'], dense=config['dense'])
else:
    raise ValueError('Invalid model name!')


# Load the data
x_train, y_train, x_val, y_val, x_test, y_test = load_data(image_folder=config['train_images_folder'],
                                                           dataset_csv=config['train_labels_csv'],
                                                           testset_csv=config['test_labels_csv'])
# Train the model
model.train(x_train, y_train, x_val, y_val, batch_size=config['batch_size'], epochs=config['epochs'])

# Evaluate the model
model.evaluate(x_test, y_test)

# Save the trained model
model.save(args.save_path)