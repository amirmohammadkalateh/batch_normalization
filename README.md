# batch_normalization
# Batch Normalization

Batch Normalization (BatchNorm) is a technique used to improve the training stability and performance of deep neural networks. It addresses the problem of internal covariate shift, which is the change in the distribution of network activations due to the parameter updates during training.

## What is Internal Covariate Shift?

Internal covariate shift refers to the change in the distribution of network activations as the parameters of the preceding layers change during training. This makes it harder for the network to learn, as each layer has to constantly adapt to the changing distribution of its inputs.

## How Batch Normalization Works

Batch Normalization normalizes the activations of a layer by subtracting the batch mean and dividing by the batch standard deviation. This is done for each mini-batch of data during training.

Specifically, for a layer with input activations `x`, Batch Normalization performs the following operations:

1.  **Calculate the batch mean (μ):**
    ```
    μ = (1/m) * Σ(x_i)
    ```
    where `m` is the batch size and `x_i` are the activations in the batch.

2.  **Calculate the batch variance (σ^2):**
    ```
    σ^2 = (1/m) * Σ((x_i - μ)^2)
    ```

3.  **Normalize the activations (x_hat):**
    ```
    x_hat = (x_i - μ) / √(σ^2 + ε)
    ```
    where `ε` is a small constant added for numerical stability.

4.  **Scale and shift the normalized activations (y):**
    ```
    y = γ * x_hat + β
    ```
    where `γ` (gamma) and `β` (beta) are learnable parameters that allow the network to scale and shift the normalized activations.

## Benefits of Batch Normalization

* **Improved Training Stability:** BatchNorm reduces internal covariate shift, leading to more stable and faster training.
* **Faster Convergence:** Networks with BatchNorm often converge faster, requiring fewer training epochs.
* **Reduced Sensitivity to Initialization:** BatchNorm makes the network less sensitive to the choice of initial weights.
* **Regularization Effect:** BatchNorm can have a slight regularization effect, reducing the need for other regularization techniques.
* **Allows Higher Learning Rates:** BatchNorm can enable the use of higher learning rates, further accelerating training.

## Implementation in TensorFlow/Keras

In TensorFlow/Keras, Batch Normalization can be easily implemented using the `tf.keras.layers.BatchNormalization` layer.

```python
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
