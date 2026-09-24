# Binary Image Preprocessing and Learning(16 Bytes, 4 Params)


This notebook implements a simple image-to-binary-mask learning pipeline using a **single 1×1 convolution layer** in TensorFlow/Keras.

The workflow converts color images into binary target images using a manually defined pixel-intensity threshold, then trains a neural network to reproduce that transformation.

---

## Project Structure

```text
project/
│
├── preprocessing_3.ipynb
├── Normal_Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── dip/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── binary.h5
└── binary predicted.png
```

---

## Requirements

The notebook uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Pillow
- TensorFlow / Keras

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow tensorflow
```

---

## 1. Loading Images

Training images are loaded from:

```text
Normal_Images/
```

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

OpenCV is used for reading the images.

---

## 2. Image Resizing

Every image is resized to:

```text
320 × 320
```

using:

```python
cv2.resize(img, (320, 320))
```

The model therefore receives:

```text
320 × 320 × 3
```

input images.

---

## 3. Pixel Normalization

The image values are divided by 255:

```python
image / 255.0
```

This converts the pixel range from approximately:

```text
0 – 255
```

to:

```text
0 – 1
```

The implementation rounds the normalized values to two decimal places.

---

# 4. Generating the Binary Target

The target image is generated manually from each normalized input image.

First, the three channel values of every pixel are averaged:

```python
pixel = np.mean(image[i, j, :])
```

A threshold of `0.6` is then applied:

```python
if pixel >= 0.6:
    pixel = 1
else:
    pixel = 0
```

Therefore:

```text
Mean pixel value >= 0.6  →  1
Mean pixel value <  0.6   →  0
```

The result is a binary image:

```text
320 × 320
```

---

## Binary Conversion Concept

```text
RGB Pixel
   │
   ▼
Mean of 3 channels
   │
   ▼
Threshold = 0.6
   │
   ├───────────────┐
   │               │
 >= 0.6          < 0.6
   │               │
   ▼               ▼
   1               0
```

This creates the `y_true` target used during training.

---

## 5. Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

The split is based on the sorted image order:

```python
train_images = resized_images[:int(len(resized_images) * 0.8)]
test_images = resized_images[int(len(resized_images) * 0.8):]
```

The same split is applied to the binary targets.

> **Note:** The split is not randomized. If the files have an ordering pattern, this can affect the representativeness of the test set.

---

# 6. Model Architecture

The notebook uses one convolutional layer:

```python
model = Sequential([
    Conv2D(
        1,
        1,
        input_shape=(320, 320, 3),
        activation='sigmoid'
    )
])
```

Architecture:

```text
Input
320 × 320 × 3
       │
       ▼
1 × 1 Conv2D
1 filter
Sigmoid activation
       │
       ▼
320 × 320 × 1
       │
       ▼
Binary Prediction
```

---

## Why 1×1 Convolution?

A 1×1 convolution processes each pixel independently while combining its input channels.

For an RGB image, the layer can learn a function similar to:

```text
z = w1 × C1 + w2 × C2 + w3 × C3 + bias
```

The sigmoid activation converts the result into a value between:

```text
0 and 1
```

This makes the architecture suitable for learning a per-pixel binary transformation.

---

# 7. Training Configuration

The model is compiled with:

- Optimizer: Adam
- Learning rate: `0.1`
- Loss: Binary Crossentropy
- Metrics:
  - Accuracy
  - MSE
  - MAE

```python
optimizer = Adam(learning_rate=0.1)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'mse', 'mae']
)
```

Training configuration:

```text
Epochs:     200
Batch size: 2
```

---

# 8. Training Monitoring

The notebook records and plots three types of measurements.

### Binary Crossentropy Loss

Shows how closely the predicted probabilities match the binary targets.

### Mean Squared Error

Measures the squared difference between prediction and target.

### Accuracy

Measures the proportion of pixels classified correctly according to the model's binary predictions/metric behavior.

The plots allow training and validation behavior to be compared over the 200 epochs.

---

# 9. Prediction

After training:

```python
pred_images = model.predict(test_images)
```

The model produces values between approximately:

```text
0 – 1
```

because the final activation is sigmoid.

For visualization, the predictions are converted to image intensity values:

```python
pred_images = pred_images * 255.0
```

Thus:

```text
0 → black
255 → white
```

The notebook displays:

1. The manually generated binary target.
2. The model's predicted image.

This gives a direct visual comparison between the target and prediction.

---

# 10. Saving the Model

The trained model is saved as:

```text
binary.h5
```

using:

```python
model.save('binary.h5')
```

The saved model can subsequently be loaded and used for inference without repeating training.

---

# Applying the Model to `dip`

The notebook then loads images from:

```text
dip/
```

and performs the same preprocessing:

```text
Read image
    ↓
Resize to 320×320
    ↓
Normalize to 0–1
    ↓
Model prediction
    ↓
Convert prediction to 0–255
    ↓
Save/display result
```

An example predicted image is saved as:

```text
binary predicted.png
```

---

# End-to-End Pipeline

```text
                    TRAINING
                       │
                       ▼
              Normal_Images
                       │
                       ▼
               Resize 320×320
                       │
                       ▼
                 Normalize
                       │
                       ▼
             Mean RGB Channels
                       │
                       ▼
             Threshold at 0.6
                       │
                       ▼
              Binary Target
                       │
                       ▼
                 80/20 Split
                       │
                       ▼
                1×1 Conv2D
                       │
                       ▼
                   Training
                       │
                       ▼
                  binary.h5
                       │
                       │
                       ▼
                   INFERENCE
                       │
                       ▼
                      dip
                       │
                       ▼
               Resize + Normalize
                       │
                       ▼
                 Trained Model
                       │
                       ▼
               Binary Prediction
                       │
                       ▼
             binary predicted.png
```

---

# Important Technical Observations

## 1. The target is threshold-based

The binary target is not manually annotated. It is generated from the input image itself using:

```text
mean(channel values) >= 0.6
```

Therefore, the neural network is learning to approximate a deterministic thresholding operation.

---

## 2. The task is pixel-wise

Because the architecture uses a 1×1 convolution, spatial neighborhoods are not considered.

For example, the value of a pixel at:

```text
(x, y)
```

does not directly depend on neighboring pixels such as:

```text
(x-1, y)
(x+1, y)
(x, y-1)
(x, y+1)
```

The decision is based on the channel values at that pixel.

---

## 3. Sigmoid output

The sigmoid activation produces a probability-like value:

```text
0 ≤ prediction ≤ 1
```

The current notebook visualizes this continuous output after multiplying it by 255.

For a strict binary output during inference, a threshold can be applied:

```python
binary_prediction = (pred_images >= 0.5).astype(np.uint8)
```

and then scaled if required:

```python
binary_prediction = binary_prediction * 255
```

---

# Possible Improvements

For a more robust implementation, consider:

1. Randomizing the train/test split.
2. Removing unnecessary rounding during normalization.
3. Making the threshold (`0.6`) a named configuration parameter.
4. Using a validation set separate from the final test set.
5. Reporting precision, recall, F1-score, IoU, and Dice where appropriate for binary masks.
6. Applying an explicit threshold to predictions before saving binary images.
7. Adding reproducibility through fixed random seeds.
8. Adding input validation for unreadable images.
9. Using reusable preprocessing and inference functions.
10. Saving the preprocessing configuration along with the model.
11. Considering the modern `.keras` format instead of legacy `.h5` where appropriate.

---

# Output Files

| File | Purpose |
|---|---|
| `binary.h5` | Trained binary prediction model |
| `binary predicted.png` | Example predicted binary image |

---

# Summary

`preprocessing_3.ipynb` demonstrates how a very small neural network can learn a pixel-level binary transformation.

The central pipeline is:

```text
3-channel image
       ↓
1×1 convolution
       ↓
Sigmoid
       ↓
Binary-like pixel prediction
```

The target is generated using a mean-channel threshold of `0.6`, making this a controlled experiment for understanding how a 1×1 convolution can learn channel-wise pixel transformations.
