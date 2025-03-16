# Depth Estimation with MiDaS in Google Colab
This guide provides step-by-step instructions to generate depth maps from single images using the MiDaS (Monocular Depth Estimation in the Wild) model within Google Colab.

## Prerequisites
A Google account to access Google Colab.
An image for which you want to generate a depth map.
## Steps

### * Open Google Colab
### * Navigate to Google Colab and create a new notebook.​
### * Install Required Libraries
In a code cell, run the following commands to install necessary libraries and clone the MiDaS repository:

```python

!pip install torch torchvision
!git clone https://github.com/isl-org/MiDaS.git
%cd MiDaS
!pip install -r requirements.txt
```
### * Import Required Libraries
Import essential libraries for image processing and depth estimation:

```python

import torch
import urllib.request
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
```
### * Load the MiDaS Model
Load the pre-trained MiDaS model for depth estimation:

``` python

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()
```
### * Define Image Transformation
Set up the transformation to preprocess the input image:

```python

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
### * Load and Preprocess the Image
Load your image and apply the defined transformations:

```python

image_url = 'https://example.com/your_image.jpg'
image_path = 'your_image.jpg'
urllib.request.urlretrieve(image_url, image_path)

img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)
```

### * Perform Depth Estimation
Use the model to predict the depth map:

```python

with torch.no_grad():
    prediction = midas(input_tensor)
```
### * Post-process and Visualize the Depth Map
Resize the prediction and display the depth map:

``` python

prediction_resized = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False,
).squeeze()

output = prediction_resized.cpu().numpy()
plt.imshow(output, cmap='plasma')
plt.axis('off')
plt.show()

```
## Notes
Replace 'https://example.com/your_image.jpg' with the URL of your image or the path to your local image file.​

## This code will generate and display a depth map for the input image using the MiDaS model in Google Colab.​

## For more details and updates, refer to the official MiDaS repository:​
GitHub: https://github.com/isl-org/MiDaS
Colab Notebook: https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/intelisl_midas_v2.ipynb
