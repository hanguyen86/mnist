# Handwritten Digit Recognition
Recognizing an handwrittent digit image using Flask framework.

## Overview
  * This project aims to demonstrate how to create a simple service-based (using Flask framework) web-application for handwritten digit recognition task (using TensorFlow framework) on MNIST dataset.

  * Repo structure:
  ```
  --app                       # web application container
      |--__init__.py            # initialize app
      |--classifier.py          # class definition for Classifier
      |--configuration.py       # contains global setting variables
      |--mnist.py               # code to run and manage Flask
      |--templates              # contains template html files
            |--404.html
            |--base.html
            |--index.html
      |--tensorflow
            |--trained_model    # contains pre-trained model
                |--cnn
                |--softmax
  --uploads                   # contains uploaded images
  ```
  * Workflow:
    - On main page, click on 'Choose File' button to select an image.
    - Click 'Submit' to upload the selected image.
    - If image is valid (PNG or JPEG format, size `28x28`), the classification result is returned in a JSON blob:
    ```json
    {
        "classifier": "SoftmaxClassifier", 
        "label": 4
    }
    ```
    - Otherwise, an 404 error will be displayed.

## Installation
1. Requires: Python 2.7, Git
2. Install pip & Virtualenv
  * Ubuntu/Linux 64-bit
  ```
  sudo apt-get install python-pip python-dev python-virtualenv
  ```
  * Mac OS X
  ```
sudo easy_install pip
sudo pip install --upgrade virtualenv
  ```
3. Create and activate Virtualenv environment
```
virtualenv --system-site-packages ~/mnist
source ~/mnist/bin/activate  # If using bash
```
4. TensorFlow: choose one of the following package
  * Ubuntu/Linux 64-bit, CPU only, Python 2.7
  ```
  export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
  ```
  * Ubuntu/Linux 64-bit, GPU enabled, Python 2.7 (Requires CUDA toolkit 8.0 and CuDNN v5)
  ```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
  ```
  * Mac OS X, CPU only, Python 2.7:
  ```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py2-none-any.whl
  ```
  * Mac OS X, GPU enabled, Python 2.7:
  ```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.0rc0-py2-none-any.whl
  ```
5. Tensorflow: install the selected package above:
  ```
pip install --upgrade $TF_BINARY_URL
  ```
6. Clone the repo and cd to mnist directory
```
git clone https://github.com/hanguyen86/mnist.git
```
7. Install Flask and other necessary packages
```
pip install -r requirements.txt
```
8. Run the server. Test the server at [http://localhost:8001/](http://localhost:8001/) 
```
python runserver.py
```

## Classification
In `classifier.py`, we defined a base class for Recognition task for *training* and *predict*. We also included 2 specific implementation using Softmax regression model and Convolution Neural Network (CNN).

`classifier.py` can be controled as an independent module.
  * For training:
  ```
  python app/classifier.py -c <Classifier_Name> -t
  ```  
  * For predicting:
  ```
  python app/classifier.py -c <Classifier_Name> -p <Image_Path>
  ```
  * Example:
  ```
  # help
  python app/classifier.py -h
  # training
  python app/classifier.py -c CNNClassifier -t
  #predicting
  python app/classifier.py -c SoftmaxClassifier -p uploads/4.png
  ```