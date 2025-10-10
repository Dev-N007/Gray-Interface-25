# Neural Style Transfer with Custom CNN

This project implements a simplified neural style transfer algorithm using TensorFlow and Keras. It blends the content of one image with the artistic style of another by optimizing a generated image to minimize content and style loss.

## Overview

Neural style transfer is a technique that applies the artistic style of one image (style image) to the content of another image (content image). This implementation uses a custom convolutional neural network (CNN) to extract features and compute losses.

## Link

https://colab.research.google.com/drive/12zktQ8BdUF-3LDXvIEShWsQ-aIKQhofl?usp=sharing

## Features

- Loads and preprocesses content and style images
- Builds a custom CNN for feature extraction
- Computes content and style loss using feature maps and Gram matrices
- Optimizes a generated image using gradient descent
- Displays intermediate and final stylized results

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Steps to Run the Code

1. **Install Required Libraries**

   If you're running this outside of Google Colab, install the necessary packages:

   """
    pip install tensorflow numpy matplotlib pillow
   """

2. **Prepare Your Images**

    Choose a content image 
    Choose a style image 
    Place both images in the appropriate directory

3.  **Open the Notebook and run all cells**

4.  **Execute the following code block at the end of the notebook:**

    """
    result = style_transfer("/content/big_ben.jpeg", "/content/starry_night.jpeg", epochs=1000, lr=0.02)
    show_image(result, "Final Stylized Image")
    """

5. **Show the output after every 50 epochs**