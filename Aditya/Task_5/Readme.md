# PlainNet-34 CNN for ImageNet-100 Classification

This project implements and trains a 34-layer plain Convolutional Neural Network (CNN), inspired by the architecture proposed in the original ResNet paper as a baseline. The model is trained on the ImageNet-100 dataset, a challenging subset of the full ImageNet dataset containing 100 classes.

The primary goal is to establish a performance baseline for a deep "plain" network, which notably lacks the residual (skip) connections that are the hallmark of ResNet architectures.

## Table of Contents
1.  [Key Features](#key-features)
2.  [Model Architecture: PlainNet-34](#model-architecture-plainnet-34)
3.  [Dataset and Preprocessing](#dataset-and-preprocessing)
4.  [Training Strategy](#training-strategy)
5.  [Results](#results)
6.  [How to Run](#how-to-run)
7.  [File Structure](#file-structure)

## Key Features

*   **34-Layer Plain CNN**: A deep convolutional network built sequentially without residual connections.
*   **ImageNet-100 Dataset**: Trained and validated on a 100-class subset of ImageNet.
*   **Data Augmentation**: Standard data augmentation techniques are used for the training set (`RandomResizedCrop`, `RandomHorizontalFlip`).
*   **Two-Phase Training**: Utilizes the Adam optimizer for the initial 30 epochs for rapid convergence, then switches to SGD with momentum for the remaining epochs to achieve finer tuning.
*   **Learning Rate Scheduling**: `StepLR` schedulers are used to decay the learning rate at fixed intervals for both optimizers.
*   **Detailed Logging**: The script automatically saves per-epoch metrics (loss, accuracy) to a CSV file and appends a final summary of the training run to a comparison file.
*   **Checkpointing**: The best model weights (based on top-1 validation accuracy) are saved automatically during training.

## Model Architecture: PlainNet-34

The model is a 34-layer CNN composed of a series of convolutional blocks. A "block" in this context consists of a `Conv2d` layer, followed by `BatchNorm2d` and a `ReLU` activation.

The architecture is defined as follows:

1.  **Initial Layer**:
    *   `7x7 Conv`, 64 filters, stride 2
    *   `BatchNorm2d`
    *   `ReLU`
    *   `3x3 MaxPool`, stride 2

2.  **conv2_x**: 6 convolutional layers (from 3 blocks) with 64 filters.
3.  **conv3_x**: 8 convolutional layers (from 4 blocks) with 128 filters. Stride is 2 in the first block to downsample.
4.  **conv4_x**: 12 convolutional layers (from 6 blocks) with 256 filters. Stride is 2 in the first block.
5.  **conv5_x**: 6 convolutional layers (from 3 blocks) with 512 filters. Stride is 2 in the first block.

6.  **Final Layers**:
    *   `AdaptiveAvgPool2d` to reduce feature maps to 1x1.
    *   `Flatten`
    *   `Linear` layer with 100 output units (for 100 classes).

**Note on "Plain" Architecture**: This network is "plain" because it consists of a simple sequential stacking of layers. It intentionally omits the shortcut or skip connections that define residual networks (ResNets). This often makes training very deep plain networks challenging due to issues like vanishing gradients.

## Dataset and Preprocessing

*   **Dataset**: [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100)
*   **Image Size**: All images are resized to `64x64` pixels.
*   **Normalization**: The dataset's mean and standard deviation are pre-calculated and used to normalize the images. The calculated values are:
    *   `mean = [0.4531, 0.4513, 0.3910]`
    *   `std = [0.1808, 0.1760, 0.1748]`
*   **Training Transforms**: `RandomResizedCrop(64)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`.
*   **Validation Transforms**: `Resize((64,64))`, `ToTensor()`, `Normalize()`.

## Training Strategy

The model is trained for **100 epochs** using a specific two-phase optimization strategy designed to balance fast initial learning with stable fine-tuning.

*   **Loss Function**: `nn.CrossEntropyLoss()`
*   **Batch Size**: 128

#### Optimizer and Scheduler Details

*   **Epochs 1-30: Adam Optimizer**
    *   Learning Rate: `1e-4`
    *   Weight Decay: `1e-4`
    *   Scheduler: `StepLR` with `step_size=10`, `gamma=0.1`. (The learning rate is reduced by a factor of 10 every 10 epochs).

*   **Epochs 31-100: SGD Optimizer**
    *   Learning Rate: `0.01`
    *   Momentum: `0.9`
    *   Weight Decay: `5e-4`
    *   Scheduler: `StepLR` with `step_size=30`, `gamma=0.1`.

The switch from Adam to SGD at epoch 31 causes a noticeable jump in training loss, as the optimizer state is reset. However, the model quickly recovers and continues to learn.

## Results

After 100 epochs of training, the model achieves the following performance on the validation set:

*   **Best Top-1 Validation Accuracy**: **22.44%** (achieved at epoch 27)
*   **Final Top-5 Validation Accuracy**: **48.88%**
*   **Total Training Time**: Approximately **15,493 seconds** (~4.3 hours) on a CUDA-enabled GPU.

The training progress, including per-epoch loss and accuracy, is logged in `results/per_model_logs/PlainNet34_ImageNet100_metrics.csv`. A summary of the final results is appended to `results/summary/final_model_comparison.csv`.

## How to Run

1.  **Prerequisites**: Ensure you have Python and the required libraries installed.
    ```bash
    pip install torch torchvision pandas numpy tqdm
    ```

2.  **Prepare the Dataset**:
    *   Download the ImageNet-100 dataset.
    *   Update the hardcoded paths in the Jupyter notebook to point to your training and validation data directories. Specifically, modify the paths in these lines:
        *   `dataset_path = r"C:\\archive\\imagenet100\\train"`
        *   `train_dataset = ImageFolder(r"C:\\archive\\imagenet100\\train", ...)`
        *   `val_dataset = ImageFolder(r"C:\\archive\\val.X", ...)`

3.  **Execute the Notebook**:
    *   Open `plain_34_layer_model.ipynb` in a Jupyter environment.
    *   Run all cells sequentially. The training process will begin and log its progress.
   
**Other Project Link**
In this seperate project I have implemented Plain architecture, VGG19, ResNet and EfficientNet models.
[Link](https://github.com/Aditya-043-nit/ImageNet100-Model-Comparison)
