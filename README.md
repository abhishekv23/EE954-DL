# Fashion-MNIST Classification with CNN and MLP

This project involves building and training a Convolutional Neural Network (CNN) combined with a Multi-Layer Perceptron (MLP) to classify images from the Fashion-MNIST dataset. The CNN extracts features from the images, which are then passed to the MLP for classification.

## Model Architectures

### CNN Model
The CNN model consists of 5 convolutional layers, each followed by ReLU activation and a max-pooling layer. The architecture is designed to extract hierarchical features from the input images.

- **Layer 1:** Conv2d (1 input channel, 16 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 2:** Conv2d (16 input channels, 32 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 3:** Conv2d (32 input channels, 64 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 4:** Conv2d (64 input channels, 128 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 5:** Conv2d (128 input channels, 256 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=1)

The final output is flattened to create feature vectors for the MLP.

### MLP Model
The MLP model takes the flattened features from the CNN and performs the classification task.

- **Layer 1:** Linear (256 input features, 256 output features), ReLU
- **Layer 2:** Linear (256 input features, 128 output features), ReLU
- **Layer 3:** Linear (128 input features, 10 output features)

## Training Setup

### Hyperparameters
- **Kernel Size:** 2
- **Number of Kernels:** [16, 32, 64, 128, 256]
- **Learning Rate:** 0.001
- **Number of Epochs:** 10

### Weight Initialization Methods
- Xavier Initialization

## Training and Evaluation

### Instructions for Running the code by exeucting each of the following code cells:

1. **Load libraries and create a class to handle csv files**
2. **Instantiate the transfromation library to Normalize  the image data**
3. **Load the Fashion MNIST training and test datasets and create pytorch data loaders**
4. **Define the MLP Model:**
   - Use the provided `MLP` class definitions.
5. **Define 'backward_pass' function**
6. **Define the Loss function**
7. **Define MLP class**
8. **Define method for MLP model training and evaluation**
9. **Set hyperparameter, instantiate MLP network and execute train_and_evaluation()**
10. **Define ploting function to visualize the result**
11. **Define CNN model**
   - Initialize the models with the desired hyperparameters and weight initialization methods.
12. **Instantiate CNN model and pass dummy date to understand the size of data at each layer:**
13. **Print the no. of parameters in CNN model**
14. **Debug code block(No need to execute)**
15. **Define Training function for CNN model**
16. **Define Evaluation and testing mothod for CNN model**
17. **Define method to visualize Loss and Accuracy**
18. **Load the pytorch data loaders**
19. **Execute CNN model training and evaluation loop using 'xavier' weight initialization method**
20. **Execute CNN model training and evaluation loop using 'he(kaiming)' weight initialization method**
21. **Execute CNN model training and evaluation loop using random weight initialization method**

     
