## Environment setup

##### Required python 3.10 

##### Run requirements.txt

`pip install -r requirements.txt`

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

Total test images **MAE** : 5.8
