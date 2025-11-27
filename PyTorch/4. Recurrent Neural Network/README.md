## 1\. Introduction to Sequence Modeling and RNNs 

The video starts by adapting a standard Fully Connected (FC) Neural Network setup for MNIST to an RNN [[00:23](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=23)].

### The Core Idea: Treating Images as Sequences

While RNNs are traditionally used for sequences like text or time-series data, they are applied here to **MNIST images** by re-framing the 28x28 pixel image as a sequence [[01:44](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=104)]:

  * **Sequence Length:** 28 time steps (one for each row of the image).
  * **Input Features:** 28 features at each time step (the 28 pixels in that row).

This approach allows the model to process the image sequentially, building context (the hidden state) as it moves from the first row to the last.

### Key Hyperparameters for RNNs

To define and train an RNN in PyTorch, you must specify four key parameters [[02:11](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=131)]:

1.  **`input_size` (28):** The number of features in the input at each time step (i.e., the number of pixels in one row).
2.  **`sequence_length` (28):** The number of time steps (i.e., the number of rows in the image).
3.  **`hidden_size` (256):** The size of the hidden state vector. This is the model's "memory" that is passed from one time step to the next.
4.  **`num_layers` (2):** The number of stacked RNN layers.



## 2\. Implementing the Basic PyTorch RNN 

The video walks through creating the `RNN` class in PyTorch, inheriting from `nn.Module` [[03:04](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=184)].

### A. Initialization (`__init__`)

1.  **Defining the RNN Layer:** The core of the model is `nn.RNN()`.
      * It takes `input_size`, `hidden_size`, and `num_layers`.
      * It also uses `batch_first=True` to specify that the input data tensor will be ordered as `(Batch_Size, Sequence_Length, Features)` [[04:29](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=269)].
2.  **Defining the Output Layer:** An FC (Linear) layer is added at the end to map the RNN output to the number of classes (10 for MNIST).
      * **Input to FC Layer:** The video initially concatenates the hidden state output from **all 28 time steps** and passes this combined vector to the FC layer [[05:37](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=337)]. The size is `hidden_size * sequence_length`.


### B. Forward Pass (`forward`)

1.  **Initializing Hidden State ($\mathbf{h_0}$):** Before the RNN can start processing the sequence, the initial hidden state must be created using `torch.zeros()` [[06:20](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=380)].
      * The shape must be `(num_layers, batch_size, hidden_size)`.
      * It must be moved to the correct device (CPU/GPU).
2.  **RNN Call:** The input tensor `X` and the initial hidden state `h0` are passed to the RNN layer: `out, h_n = self.rnn(X, h0)` [[07:01](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=421)].
      * **`out`:** Contains the hidden state output for **every time step**. Shape is `(Batch, Sequence, Hidden_Size)`.
      * **`h_n`:** Contains the **final hidden state** for all layers.
3.  **Reshaping and Output:** The output tensor `out` is reshaped (flattened) to combine the sequence and hidden dimensions, preparing it for the final linear classification layer [[07:34](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=454)]. The result is the prediction logits.

### C. Data Preprocessing (Crucial Fix)

A common mistake in PyTorch is incorrect input tensor shape. The MNIST data is loaded as `(N, 1, 28, 28)`. The RNN expects `(N, 28, 28)`. The fix is to use `data.squeeze(1)` to remove the redundant channel dimension (the '1') before feeding it to the network [[08:38](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=518)].


## 3\. Advanced Recurrent Architectures 

The video shows how to easily switch from a basic RNN to more advanced, more capable architectures like GRUs and LSTMs, which solve the **vanishing gradient problem** and better capture long-range dependencies.

### A. Gated Recurrent Unit (GRU)

The GRU is implemented simply by replacing `nn.RNN` with **`nn.GRU`** [[10:04](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=604)].

  * **Mechanism:** The GRU uses **Update** and **Reset** gates to control the flow of information through the hidden state. It is simpler and has fewer parameters than the LSTM.
  * **Implementation:** No other code changes are needed because the GRU, like the basic RNN, only uses a single hidden state ($\mathbf{h}$) and does not require a separate cell state.

### B. Long Short-Term Memory (LSTM)

The LSTM is slightly more complex than the GRU [[10:37](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=637)].

  * **Mechanism:** LSTMs introduce a separate **Cell State ($\mathbf{c}$)**, along with **Forget, Input, and Output gates**. This cell state acts as a long-term memory, allowing information to be preserved over very long sequences. \* **Implementation:**
    1.  Replace `nn.RNN` with **`nn.LSTM`**.
    2.  In the `forward` method, an initial **Cell State ($\mathbf{c_0}$)** must be created using `torch.zeros()` alongside the hidden state $h_0$ [[10:52](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=652)].
    3.  The LSTM layer call now requires both the hidden state and cell state as a tuple: `out, (h_n, c_n) = self.lstm(X, (h0, c0))` [[11:22](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=682)].


## 4\. Alternate Output Strategy: Using the Last Hidden State 

The video highlights an alternative approach for getting the final classification output [[12:11](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=731)].

### Concatenation (Original Strategy)

  * **Method:** Concatenate the hidden states from **all** time steps and feed this massive vector to the FC layer.
  * **Pros:** Uses all information generated at every step.
  * **Cons:** High dimensionality, potentially redundant information.

### Last Hidden State (Alternative Strategy)

  * **Method:** Only use the hidden state from the **final time step ($\mathbf{h_{T}}$)** for classification.
  * **Justification:** In theory, the final hidden state should contain a distilled representation of all prior information in the sequence.
  * **Implementation:** The complex reshaping/concatenation step is removed. Instead, the model extracts the last element of the sequence dimension from the output tensor `out` [[12:49](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=769)]:

<!-- end list -->

```python
# 'out' originally has shape (Batch, Sequence_Length, Hidden_Size)
# We select only the last step (-1) from the sequence dimension (1)
out = out[:, -1, :]
```

This reduces the input to the final FC layer from `(Batch, Sequence_Length * Hidden_Size)` to simply `(Batch, Hidden_Size)`. Surprisingly, in this example, using only the last hidden state improved the accuracy [[13:32](http://www.youtube.com/watch?v=Gl2WXLIMvKA&t=812)].


## 5\. Summary of Results (MNIST) 

The video summarizes the performance of each model on the MNIST classification task:

| Model | Accuracy (Approx.) | Note |
| :--- | :--- | :--- |
| Basic RNN | \~97.28% | Serves as a baseline. |
| GRU | \~98.10% | Shows improvement over the basic RNN, capturing better dependencies. |
| LSTM | \~97.90% | Comparable to GRU, often the default choice in practice. |
| LSTM (Last Hidden State) | **\~98.40%** | Achieved the best result in this specific experiment, demonstrating that capturing the sequence's final state can sometimes be sufficient or superior. |

http://googleusercontent.com/youtube_content/8
