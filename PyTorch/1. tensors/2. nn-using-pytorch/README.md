## 1\. Setting Up the PyTorch Environment and Hyperparameters 

The first essential step in any PyTorch project is setting up the coding environment and defining the initial parameters that control the training process.

### Core PyTorch Imports (The Toolbox)

The video imports several key modules:

| Module | Purpose | In-Depth Explanation |
| :--- | :--- | :--- |
| `torch` | The main PyTorch library. | Used for all tensor operations (arrays), the fundamental building block of all data and computations. |
| `torch.nn` (as `nn`) | Neural Network Modules. | Contains all the core classes for building networks, such as layers (`nn.Linear`), activation functions (though often called via `F`), and loss functions (`nn.CrossEntropyLoss`). All custom models inherit from `nn.Module`. |
| `torch.optim` | Optimization Algorithms. | Holds algorithms like **Adam** and **Stochastic Gradient Descent (SGD)**, which adjust the model's weights to minimize the loss. |
| `torch.nn.functional` (as `F`) | Functional Components. | Contains functions that don't have learnable parameters, primarily used for activation functions like **ReLU** and **Softmax** within the `forward` method. |
| `torch.utils.data` | Data Utilities. | Provides the `DataLoader` class, crucial for managing datasets, creating mini-batches, and shuffling data for training. |
| `torchvision.datasets` | Standard Datasets. | PyTorch's library for computer vision, offering easy access to standard datasets like **MNIST** (used here). |
| `torchvision.transforms` | Data Transformations. | Provides tools to preprocess data, most notably `transforms.ToTensor()`, which converts image data (like NumPy arrays or PIL images) into PyTorch tensors. |

### Device Initialization (CPU vs. GPU)

The code initializes the `device` variable to specify where computations will run:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

  * **`cuda`:** Refers to an NVIDIA GPU. If a compatible GPU is found (`torch.cuda.is_available()` is true), training runs much faster.
  * **`cpu`:** Used as a fallback if no GPU is available. All model parameters and data tensors must be moved to this device for computation.

### Hyperparameters

The key hyperparameters for this network are set:

  * **`input_size` (784):** The total number of features for a single input image. Since MNIST images are 28x28 pixels, they are flattened: $28 \times 28 = 784$.
  * **`num_classes` (10):** There are 10 possible digit classes (0 through 9). The output layer must have 10 nodes.
  * **`learning_rate` (0.001):** Controls the step size of the optimizer during weight updates.
  * **`batch_size` (64):** The number of data samples processed together in one forward/backward pass. Using batches (mini-batches) makes training more efficient and stable.
  * **`num_epochs` (1):** The number of times the entire dataset will be passed through the network.



## 2\. Defining the Neural Network Architecture 

A PyTorch network is defined as a Python class that inherits from `nn.Module`. This custom class contains the architecture of the model.

### The `__init__` Method (The Structure)

This method defines the layers and components of the network.

  * `super().__init__()`: Calls the constructor of the parent class (`nn.Module`), which is necessary for PyTorch to track the model's parameters and methods correctly.
  * **Layers:** The model uses a simple two-layer FCNN:
    1.  `self.fc1 = nn.Linear(input_size, 50)`: The first **Linear** (fully connected) layer maps the 784 input features to 50 hidden nodes.
    2.  `self.fc2 = nn.Linear(50, num_classes)`: The second Linear layer maps the 50 hidden nodes to the 10 output classes.

### The `forward` Method (The Calculation)

This method defines the sequence of operations (the forward pass) that the input data (`x`) goes through to produce the output scores.

1.  **First Layer:** `x = self.fc1(x)` - Input is passed through the first linear layer.
2.  **Activation:** `x = F.relu(x)` - The **Rectified Linear Unit (ReLU)** activation function is applied. This introduces non-linearity, allowing the network to learn complex patterns.
3.  **Second Layer:** `x = self.fc2(x)` - The result is passed through the final linear layer, which outputs the raw scores (logits) for each of the 10 classes.
4.  **Return:** The raw scores are returned. These scores represent the model's confidence that the input belongs to each of the 10 classes.

> **Key Concept: Shape Check**
> The video performs a shape check with dummy data to ensure the model's output shape is correct (`[Batch Size, Num Classes]`, e.g., `[64, 10]`), confirming that the network is structured correctly before training begins.



## 3\. Data Loading and Preprocessing 

For the model to learn, it needs data. The video uses the **MNIST dataset**, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

### Dataset and Transformation

1.  **Loading the Dataset:** `datasets.MNIST(...)` loads the data. The `train=True` argument specifies the training set, and `download=True` ensures the data is fetched if it doesn't exist locally.
2.  **`transforms.ToTensor()`:** This is a crucial preprocessing step. It converts the pixel values from an integer range (0-255) into a floating-point tensor range (0.0-1.0). This normalization is vital for stable and effective neural network training.

### The `DataLoader`

The `DataLoader` is the abstraction that makes training practical:

```python
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

  * **Mini-Batching:** It groups the dataset into mini-batches of size 64 (the `batch_size`). The network computes gradients and updates weights after processing *each batch*, not the entire dataset, which is faster and more memory-efficient.
  * **Shuffling:** `shuffle=True` ensures the data order is randomized at the start of each epoch, preventing the network from learning spurious ordering patterns in the dataset.
  * **Training vs. Testing:** Separate loaders are created for the training data and the test data.

-----

## 4\. Training Setup: Loss Function and Optimizer 🛠️

Before entering the training loop, the final components for calculating error and updating weights are defined.

### Loss Function (Criterion)

```python
criterion = nn.CrossEntropyLoss()
```

  * **Function:** **Cross-Entropy Loss** is the standard loss function for multi-class classification problems like MNIST.
  * **Role:** It measures the difference (error) between the model's predicted raw scores (logits) and the true label (target). A lower loss value means better performance.

### Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

  * **Algorithm:** The **Adam** optimizer is used. It is an extension of Stochastic Gradient Descent that uses adaptive estimates of moments to adjust the learning rate for each weight individually.
  * **Parameters:** It takes `model.parameters()` as input, which tells the optimizer *which* parameters (weights and biases) in the network it needs to update during the backward pass.
  * **Learning Rate (`lr`):** The `learning_rate` is passed to the optimizer, determining the magnitude of the weight adjustments.



## 5\. The Training Loop (The Learning Process) 

The core of the tutorial is the training loop, which runs for a specified number of epochs.

### Epoch and Batch Iteration

The training process involves two nested loops:

1.  **Outer Loop (`for epoch in range(num_epochs)`):** Iterates over the entire dataset once per epoch.
2.  **Inner Loop (`for batch_idx, (data, targets) in enumerate(train_loader)`):** Iterates over all the mini-batches within a single epoch.

### Six Steps of the Backward Pass

Inside the inner loop, a standardized sequence of six steps occurs for every batch:

1.  **Move to Device:** `data = data.to(device)` and `targets = targets.to(device)` are called to ensure the data is on the correct device (GPU or CPU) for computation.
2.  **Reshape/Flatten:** `data = data.reshape(data.shape[0], -1)` converts the 2D image data (28x28) into a 1D vector (784), which is required by the first `nn.Linear` layer.
3.  **Zero Gradients:** `optimizer.zero_grad()` must be called to reset all the gradients from the previous batch. PyTorch accumulates gradients by default, so they must be explicitly zeroed out for each new iteration.
4.  **Forward Pass:** `scores = model(data)` runs the input data through the network's `forward` method to get the predicted scores.
5.  **Calculate Loss:** `loss = criterion(scores, targets)` calculates the error (loss) based on the predicted scores and the true labels.
6.  **Backward Pass (Backpropagation):** `loss.backward()` computes the gradient (derivative) of the loss with respect to every single parameter in the model.
7.  **Update Weights (Optimizer Step):** `optimizer.step()` uses the computed gradients to adjust the network's weights and biases according to the Adam optimization algorithm.


## 6\. Evaluating Performance (Check Accuracy) 

After training, a separate function is defined to evaluate the model's performance on both the training and test datasets.

### Evaluation Mode and No Gradient

Two essential commands are used to prepare the model for evaluation:

1.  **`model.eval()`:** Sets the model to evaluation mode. This is critical because certain layers (like Dropout or BatchNorm) behave differently during training vs. evaluation, and this command disables their training-specific behavior.
2.  **`with torch.no_grad():`:** This context manager temporarily disables PyTorch's gradient tracking. Since you are only checking accuracy and not updating weights, there's no need to compute and store gradients, which saves significant memory and computation time.

### Prediction and Accuracy Calculation

Inside the evaluation loop, the model runs a forward pass, and the scores are converted to a final prediction:

1.  **Prediction:** `predictions = scores.max(1).indices`
      * `scores.max(1)` finds the maximum score along dimension 1 (the 10 class scores).
      * `.indices` extracts the index of that maximum score (e.g., index 5 for the digit '5'), which is the model's final predicted class.
2.  **Correct Count:** `num_correct` sums up how many predictions match the true labels (`predictions == y`).
3.  **Final Accuracy:** The total number of correct predictions is divided by the total number of samples processed (`num_samples`) to get the final accuracy percentage.

The video concludes by demonstrating that even a single epoch on a small network can achieve a respectable accuracy (around 93%) on the MNIST dataset, proving the entire workflow is successful.

http://googleusercontent.com/youtube_content/4
