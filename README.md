# U-Net for Image Segmentation

This repository contains an implementation of the U-Net architecture for image segmentation tasks, specifically for segmenting cell images. The U-Net model is a convolutional neural network that has been widely used for biomedical image segmentation tasks.

## Model Architecture

The U-Net model consists of two main parts: an encoder and a decoder. The encoder is responsible for capturing the context and extracting features from the input image, while the decoder is responsible for precise localization and segmentation using the encoded features.

The encoder consists of a series of convolutional blocks, followed by max-pooling operations, which progressively downsample the input image. The decoder, on the other hand, consists of upsampling operations followed by convolutional blocks, which reconstruct the segmentation map from the encoded features.

The architecture also includes skip connections between the encoder and decoder, allowing for the concatenation of low-level features from the encoder with the upsampled output of the decoder. This helps the model retain fine-grained information and improve the segmentation performance.

## Usage

1. **Data Preparation**: The code assumes that you have a dataset of images and corresponding segmentation masks. The paths to these files should be provided as lists (`image_files` and `masks_files`).

2. **Dataset and Dataloader**: The `CustomDataset` class is provided to load and preprocess the images and masks. It also applies data augmentation techniques like rotation, flipping, and normalization. The `DataLoader` is used to create batches of data for training and evaluation.

3. **Training**: The `train_loop` function is responsible for training the model. It includes early stopping and visualization of training and validation losses. The `evaluate` function is used to evaluate the model on the validation set during training.

4. **Visualization**: The `printing_images` function is provided to visualize the input images, ground truth masks, and predicted segmentation masks during training or evaluation.

5. **Model Saving**: After training, the model can be saved using `torch.save`.

## Requirements

- Python 3.x
- PyTorch
- Albumentations
- NumPy
- Pillow
- Matplotlib
- OpenCV

## Example Usage

```python
# Load the model
model = UNET(3, 64, 1, padding=0, downhill=4).to(DEVICE)

# Set up the optimizer, loss function, and gradient scaler
optim = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

# Train the model
train_loop(model, train_dataloader, val_dataloader, optim, loss_fn, scaler, EPOCHS, PATIENCE)

# Save the model
torch.save(model, 'unet_cells.pth')
```

Feel free to modify and adapt the code to your specific use case and dataset.
