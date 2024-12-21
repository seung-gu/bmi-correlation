## Environment setup

##### Required python 3.10 

##### Run requirements.txt

`pip install -r requirements.txt`

##### Setup .env file
```
TRAIN_IMAGES_FOLDER=${your_train_images_folder}
TRAIN_LABELS_CSV=${your_train_labels_csv}
TEST_LABELS_CSV=${your_test_labels_csv}
```

## Dataset load

There are two different model versions (_resnet50v2_, _vgg16_).

You can set dataset path (images, training_label, test_label) under **dataset** folder and training parameters under **models/${model_name}.json** file ({model_name}: resnet50v2, vgg16)

## Training

run
`python trainig.py --model ${model_name} --save_path ${your_model_save_path}`


## Inference

run
`python3 inference.py --model_path ${model_path} --test_results ${your_test_result_path}`

![image](https://github.com/user-attachments/assets/f3bb2d84-a57a-41cc-89fe-d7cd9bc5b6cd)

## Results

**MAE** of the total test images 

VGG16 : **5.09**

ResNet50v2 : **5.80**

VGG16 outperforms ResNet50v2 in terms of MAE, but ResNet50v2 is less complex and faster than VGG16.


## Reference
_**research.ipynb**_ (Colab file)

It was trained in many different models and parameters to find the best model and parameters.

Tested models: **VGG16**, **VGG19**, **ResNet50**, **ResNet50v2**, **InceptionV3**

## Dataset download

### Dataset
https://drive.google.com/file/d/1KbkAfwbGdxskWQoxyy4bK-fwtDgFrL4G/view?usp=drive_link

### Fine-tuned models
[ResNet50v2]

https://drive.google.com/file/d/153Zi2oIjApnbj3EDe1ME_ZDSR1YV3uft/view?usp=sharing

[VGG16]

https://drive.google.com/file/d/1-1dgSZ_j_x_DO86KqnWan54OVD1Uw_gr/view?usp=drive_link
