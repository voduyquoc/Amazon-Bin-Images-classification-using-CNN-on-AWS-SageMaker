# Inventory Monitoring at Distribution Centers using Convolutional Neural Network on SageMaker

## Project Overview
Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. In this project, we will build a model that can count the number of objects in each bin. We are supposed to count every object instances in the bin. We count individual instances separately, which means if there are two same objects in the bin, we count them as two. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.

To build this project, we will use AWS SageMaker and good machine learning engineering practices to fetch data from a database, pre-process it, and then train a machine learning model. This project will serve as a demonstration of end-to-end machine learning engineering skills that I have learned as a part of Udacity AWS Machine Learning Engineer nanodegree.

## Dataset
The dataset that will be used in this project is [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/). This dataset will be obtained from AWS. The dataset contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object. 

## Task
For this project, we will try to classify the number of objects in each bin based on a small subset of Amazon Bin Image Dataset (10,000 images). We use this subset to prevent any excess SageMaker credit usage from Udacity.

## Libraries used in the project
1. **pytorch** - used for training deep learning CNN
2. **PIL** - used for image processing
3. **Pandas, NumPy** - used for data processing
4. **Matplotlib** - used for data visualization
5. **shutil** - used for copying files

## Project Set Up and Installation
This project can be performed in any of the three softwares: **AWS Sagemaker Studio, Jupyter Lab/ Notebooks, or Google Colab**. Open the "sagemaker.ipynb" file and start by installing all the dependencies. For ease of use we may want to use a Kernel with GPU so that the training process is quick and time saving. 

## Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

![img1](https://github.com/voduyquoc/Amazon-Bin-Images-classification-using-CNN-on-AWS-SageMaker/blob/main/Snapshots/01.png)

## Overview of Project Steps

The jupyter notebook "sagemaker.ipynb" walks through implementation of Image Classification Machine Learning Model to classify the number of objects in each bin (5 classes) using the Amazon Bin Image Dataset (https://registry.opendata.aws/amazon-bin-imagery/)

- We will be using a pre-trained ResNet-50 model from pytorch vision library (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)
- We will be adding in one Fully connected Neural network Layer on top of the above ResNet-50 model.
- Note: We will be using concepts of Transfer learning and so we will be freezing all the existing Convolutional layers in the pre-trained ResNet-50 model and only changing gradients for the one fully connected layer that we have added.
- Then we will perform Hyperparameter tuning, to help figure out the best hyperparameter to be used for our model.
- Next we will be using the best hyperparameter and fine-tuning our ResNet-50 model.
- We will also be adding in configuration for Profiling and Debugging our training mode by adding in relevant hooks in the Training and Testing (Evaluation) phases.
- Next we will be deploying our model. While deploying, we will create our custom inference script. The custom inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.
- Finally we will be testing out our model with some test images of bins, to verify if the model is working as per our expectations.

## Files Used

- hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and validate the models with different hyperparameters to find the best hyperparameter.
- train_model.py - This script file contains the code that will be used by the training job to train and test the model with the best hyperparameters that we got from hyperparameter tuning.
- endpoint_inference.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations), serialization - deserialization, predictions/inferences and post-processing using the saved model from the training job.
- sagemaker.ipynb -- This jupyter notebook contains all the code and steps that we performed in this project and their outputs.

## Hyperparameter Tuning

- The ResNet-50 model with a Fully connected Linear Neural Network layer's is used for this image classification problem. ResNet-50 is 50 layers deep and is trained on a million images of 1,000 categories from the ImageNet database. Furthermore the model has a lot of trainable parameters, which indicates a deep architecture that makes it better for image recognition
- The optimizer that we will be using for this model is AdamW ( For more info refer : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )
- Hence, the hyperparameters selected for tuning were:
  - Learning rate - default(x) is 0.001 , so we have selected 0.01x to 100x range for the learning rate
  - eps - default is 1e-08 , which is acceptable in most cases so we have selected a range of 1e-09 to 1e-08
  - Weight decay - default(x) is 0.01 , so we have selected 0.1x to 10x range for the weight decay

### HyperParameter Tuning Job
![img2]

### Multiple training jobs triggered by the HyperParameter Tuning Job
![img3]

### Best hyperparameter Training Job
![img4]

### Best hyperparameter Training Job Logs
![img5]

## Debugging and Profiling
We had set the Debugger hook to record and keep track of the Loss Criterion metrics of the process in training and testing phases. The Plot of the Cross entropy loss is shown below:
![img6]

There is anomalous behavior of not getting smooth output lines.

- How would I go about fixing the anomalous behavior?
  - Making some adjustments in the pre-trained model to use a different set of the fully connected layers network, ideally should help to smoothen out the graph.
  - If I had more AWS credits, then would have changed the fc layers used in the model. Firstly would try by adding in one more fc layer on top of the existing two layers and check the results, and then if the results didn't improve much then would try by removing all the fc layers and keeping only one fc layer in the model and then rerun the tuning and training jobs and check the outputs

## Endpoint Metrics
![img7]

## Model Deployment
- Model was deployed to a "ml.t2.medium" instance type and we used the "endpoint_inference.py" script to setup and deploy our working endpoint.
- For testing purposes , we will be using some test images that we have stored in the "testImages" folder.
- We will be reading in some test images from the folder and try to send those images as input and invoke our deployed endpoint
- We will be doing this via two approaches
  - Firstly using the Predictor class object
  - Secondly using the boto3 client

## Deployed Active Endpoint Snapshot
![img8]

## Sample output returned from endpoint Snapshot
![img9]
