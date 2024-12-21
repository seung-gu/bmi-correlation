import argparse
import random
import matplotlib.pyplot as plt # Import matplotlib.pyplot
import matplotlib
import numpy as np
import os
from tensorflow.keras.models import load_model  # Import load_model from tensorflow.keras.models
from load_dataset import load_test_data  # Import x_test, y_test, and test_filenames from load_dataset.py
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()
matplotlib.use('Agg')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Infer a model.')
parser.add_argument('--model_path', type=str, required=True, help='Model path to load')
parser.add_argument('--test_results', type=str, required=True, help='Path to save the test results')
args = parser.parse_args()


test_images, test_texts, test_filenames = load_test_data(image_folder=os.getenv('TRAIN_IMAGES_FOLDER'),
                                                         testset_csv=os.getenv('TEST_LABELS_CSV'))

model = load_model(args.model_path)  # Load the model
result_folder = args.test_results

# Make predictions on test data
predictions = model.predict(test_images)

# Calculate MAE
mae = mean_absolute_error(test_texts, predictions.flatten())
print(f"Mean Absolute Error (MAE) of the total test images: {mae:.2f}")

# Visualize predictions and ground truth
num_samples_to_display = 5  # Number of samples to visualize
random_indices = random.sample(range(len(test_images)), num_samples_to_display)

# Initialize a list to store MAE for each selected image
mae_list = []
for i in random_indices:
    filename_without_extension = os.path.basename(test_filenames[i]).split(".")[0]
    index = np.where(test_filenames == filename_without_extension + ".bmp")[0][0]

    image = test_images[i]
    predicted_text = predictions[i][0]  # Assuming single output for regression
    ground_truth_text = test_texts[index]
    filename = test_filenames[index]  # Get the filename

    # Calculate MAE for the current image
    mae = mean_absolute_error([ground_truth_text], [predicted_text])
    mae_list.append(mae)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_text:.2f} \n Filename: {filename}")  # Format for regression output

    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, ground_truth_text, ha='center', va='center', fontsize=12)
    plt.title("Ground Truth")
    plt.axis('off')

    plt.savefig(f'{result_folder}/output_{i}.png')  # Save the figure as a file
    plt.close()  # Close the figure after saving it

# Calculate the average MAE
average_mae = np.mean(mae_list)
print(f"Average Mean Absolute Error (MAE) for selected images: {average_mae:.2f}")