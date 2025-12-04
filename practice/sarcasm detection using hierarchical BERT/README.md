## 1. Project Goal and Data Set

### Project Goal: Binary Classification
The project aims to perform **sarcasm detection** [[00:00](http://www.youtube.com/watch?v=63O81OUNY_g&t=0)]. This is framed as a **binary classification problem** [[01:17](http://www.youtube.com/watch?v=63O81OUNY_g&t=77)]:
* **Label 0:** Non-sarcastic comment.
* **Label 1:** Sarcastic comment.

### Data Set Used
The model is trained on a large public data set: **Sarcasm on Reddit (SARC)** [[01:08](http://www.youtube.com/watch?v=63O81OUNY_g&t=68)].
* It is a large-scale data set originally containing **1.3 million labeled comments** [[01:13](http://www.youtube.com/watch?v=63O81OUNY_g&t=73)].
* To simplify the training process and reduce computation time, the video's implementation only uses the **first 10,000 rows** of the data set [[05:18](http://www.youtube.com/watch?v=63O81OUNY_g&t=318)].
* The two critical columns used are: **`comment`** (the input text) and **`label`** (the target) [[04:50](http://www.youtube.com/watch?v=63O81OUNY_g&t=290)].

***

## 2. Data Preprocessing and Tokenization

Effective Natural Language Processing (NLP) requires meticulous cleaning and preparation of the raw text data.

### Data Cleaning and Preparation
1.  **Handle Null Values:** The single null value found in the 10,000 rows is dropped, as it has a negligible impact on a data set of this size [[06:40](http://www.youtube.com/watch?v=63O81OUNY_g&t=400)].
2.  **Remove Numerals and Symbols:** Regular expressions (`re`) are used to replace any characters that are *not* standard letters (A-Z, a-z) with a space. This is done because numerals and symbols are deemed irrelevant to sarcasm detection [[07:17](http://www.youtube.com/watch?v=63O81OUNY_g&t=437)].
3.  **Convert to Lowercase:** All text is converted to lowercase. While the chosen BERT model (Uncased) handles mixed-case data, this is a standard and recommended practice in text preprocessing [[09:16](http://www.youtube.com/watch?v=63O81OUNY_g&t=556)].

### Tokenization with BERT
**Tokenization** is the process of breaking down text into discrete units (tokens) that the model can understand.

1.  **Import BERT Tokenizer:** The `BertTokenizer` is imported from the `transformers` library [[11:35](http://www.youtube.com/watch?v=63O81OUNY_g&t=695)].
2.  **Model Type:** The `bert-base-uncased` pre-trained model is used [[12:00](http://www.youtube.com/watch?v=63O81OUNY_g&t=720)]:
    * **`base`:** Refers to the smaller BERT model with 12 encoder layers, suitable for this size of data set.
    * **`uncased`:** Means the model treats 'Hello' and 'hello' as the same.
3.  **Tokenization Parameters:** The tokenizer applies crucial NLP techniques [[14:44](http://www.youtube.com/watch?v=63O81OUNY_g&t=884)]:
    * **Max Length (100):** Every input sentence is fixed to a length of 100 tokens.
    * **Truncation:** If a comment is longer than 100 tokens, the excess tokens are discarded.
    * **Padding:** If a comment is shorter than 100 tokens, special **padding tokens (0s)** are added to the end to meet the max length requirement.
4.  **Outputs and Train-Test Split:** The tokenization produces two main arrays [[17:16](http://www.youtube.com/watch?v=63O81OUNY_g&t=1036)]:
    * **`input_ids`:** The tokenized numerical representation of the text.
    * **`attention_mask`:** A binary mask (1s for real tokens, 0s for padded tokens) that tells the model which tokens to pay attention to.
    * The model uses the **`input_ids`** as the feature set ($\mathbf{X}$) and the **`label`** column as the target ($\mathbf{Y}$) for the standard **80/20 train-test split** [[20:25](http://www.youtube.com/watch?v=63O81OUNY_g&t=1225)].

***

## 3. The Hierarchical BERT (H-BERT) Architecture

The key to this project is the complex, layered model design, which combines different strengths of deep learning architectures to understand the subtle cues of sarcasm.

The architecture is built using the Keras `Model` subclassing approach in TensorFlow [[26:10](http://www.youtube.com/watch?v=63O81OUNY_g&t=1570)].

### Layer 1: BERT Embeddings (Foundation)
* The pre-trained **`TF_BertModel`** is loaded [[45:10](http://www.youtube.com/watch?v=63O81OUNY_g&t=2710)].
* Its primary function is to transform the raw token `input_ids` into **rich, context-aware vector representations** (embeddings) [[35:31](http://www.youtube.com/watch?v=63O81OUNY_g&t=2131)].

### Layer 2: Sentence Encoding Layer (Dense Network)
* The raw BERT embeddings are passed through a dense (fully connected) layer [[28:48](http://www.youtube.com/watch?v=63O81OUNY_g&t=1728)].
* **Purpose:** To encode the input data into a **fixed-size vector** (768 units for BERT base), preparing the sequence for later layers.

### Layer 3: Context Summarization Layer (Global Average Pooling 1D)
* A **Global Average Pooling 1D** layer is applied [[29:30](http://www.youtube.com/watch?v=63O81OUNY_g&t=1770)].
* **Purpose:** To combine the sequence of individual token embeddings into a **single, summarizing vector** [[24:43](http://www.youtube.com/watch?v=63O81OUNY_g&t=1483)]. This aggregates the information from the entire sequence into a compact representation, reducing the dimensionality.

### Layer 4: Context Encoder Layer (Bidirectional LSTM)
* The single context vector's dimension is first **expanded** to add a "time step" dimension, a necessary format for LSTM layers [[38:10](http://www.youtube.com/watch?v=63O81OUNY_g&t=2290)].
* A **Bidirectional LSTM** (Long Short-Term Memory) is used [[30:37](http://www.youtube.com/watch?v=63O81OUNY_g&t=1837)].
* **Purpose:** LSTM is excellent at capturing **temporal dependencies** (sequential order) in the data. Bidirectional processing means the LSTM analyzes the text **forward and backward** [[23:24](http://www.youtube.com/watch?v=63O81OUNY_g&t=1404)], which helps the model understand the entire context of a sequence, critical for identifying linguistic contradiction often present in sarcasm.

### Layer 5: Feature Extraction Layer (CNN 1D)
* After the LSTM, the dimension is **squeezed** back, and a **channel dimension** is explicitly added to fit the CNN input format [[40:08](http://www.youtube.com/watch?v=63O81OUNY_g&t=2408)].
* A **Convolutional 1D (Conv1D) layer** is used, followed by a **Global Max Pooling 1D** layer [[32:10](http://www.youtube.com/watch?v=63O81OUNY_g&t=1930)].
    * ***Note on 1D vs. 2D:*** The original research paper used Conv2D due to their enormous data set (1.3 billion rows). The video uses **Conv1D** because the 10,000-row sample is not large enough for a 2D convolution to be effective [[25:20](http://www.youtube.com/watch?v=63O81OUNY_g&t=1520)].
* **Purpose:** CNNs are used here to **extract local features** (e.g., short phrases or patterns) from the sequence, emphasizing the most **significant features** [[23:35](http://www.youtube.com/watch?v=63O81OUNY_g&t=1415)].

### Layer 6 & 7: Final Layers
1.  **Fully Connected Layer (Dense):** Used to combine and interpret the complex features extracted by the previous layers [[33:35](http://www.youtube.com/watch?v=63O81OUNY_g&t=2015)].
2.  **Output Layer (Dense):** A final dense layer with only **one unit** and a **Sigmoid** activation function [[34:00](http://www.youtube.com/watch?v=63O81OUNY_g&t=2040)].
    * **Purpose:** Since this is binary classification, the Sigmoid function outputs a probability (a value between 0 and 1) representing the likelihood that the input is sarcastic.

***

## 4. Model Training and Evaluation

### Compilation and Training
* **Optimizer:** **Adam** is used—a popular, robust, and highly effective optimization algorithm [[48:00](http://www.youtube.com/watch?v=63O81OUNY_g&t=2880)].
* **Loss Function:** **Binary Cross-Entropy** is used, the standard loss function for binary classification problems [[48:10](http://www.youtube.com/watch?v=63O81OUNY_g&t=2890)].
* **Metrics:** The model is evaluated based on **Accuracy** [[48:21](http://www.youtube.com/watch?v=63O81OUNY_g&t=2901)].
* **Training (`model.fit`):** The model is trained over **5 epochs** with a **batch size of 32** [[48:47](http://www.youtube.com/watch?v=63O81OUNY_g&t=2927)].

### Results and Interpretation
* After training, the final accuracy achieved on the test set is approximately **63%** [[50:12](http://www.youtube.com/watch?v=63O81OUNY_g&t=3012)].
* The video notes that while this accuracy might seem low, it is considered **good** given the complexity of the task (sarcasm detection is difficult) and the non-traditional nature of the Hierarchical BERT approach, demonstrating its viability [[50:27](http://www.youtube.com/watch?v=63O81OUNY_g&t=3027)].



http://googleusercontent.com/youtube_content/12
