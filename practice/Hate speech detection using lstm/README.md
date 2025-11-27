## 1. Project Overview and Core Concepts

The project implements **Hate Speech and Offensive Language Detection**, a classic task in **Natural Language Processing (NLP)** [[00:07](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=7)]. Since text data is a sequence of words, a type of **Recurrent Neural Network (RNN)**, specifically **LSTM**, is chosen for its ability to remember dependencies over long sequences of data.

* **Hate Speech Detection:** This is a **multi-class classification** problem where a given tweet is classified into one of three categories:
    * **0: Hate Speech**
    * **1: Offensive Language**
    * **2: Neither** (Normal/Neutral)
* **Long Short-Term Memory (LSTM):** This is an advanced type of RNN architecture designed to overcome the **vanishing gradient problem** prevalent in standard RNNs. LSTMs are perfect for sequence data because they contain **memory cells** and **gates** (**input, forget, and output**) that control the flow of information, allowing the network to selectively remember or forget past information over long periods .


## 2. Data Cleaning: Preparing Raw Tweets for NLP

Before any NLP or modeling can begin, the raw data must be cleaned. The initial dataset has 24,783 rows and 7 columns [[01:25](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=85)].

### Initial Data Setup [[01:39](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=99)]

1.  **Column Selection:** The majority of columns are dropped, keeping only the **`Tweet`** column (the text input) and the **`Class`** column (the target label) [[01:45](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=105)].
2.  **Null Check:** A check confirms there are no missing (`null`) values in the remaining data columns [[03:17](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=197)].

### Regular Expression (RegEx) Cleaning [[05:07](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=307)]

Raw tweets contain unwanted characters like symbols, numbers, and multiple spaces, which confuse the model.

1.  **Symbol and Numeric Deletion:**
    * A new column, `processed_tweet`, is created.
    * **RegEx** is used to replace any character that is **NOT** a lowercase alphabet (`a-z`) or an uppercase alphabet (`A-Z`) with a single space [[05:51](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=351)].
    * The RegEx pattern used is `[^a-zA-Z]`.
2.  **Handling Unwanted Spaces:**
    * After the first step, multiple unwanted spaces often appear where symbols were deleted.
    * A second column, `processed_tweet_2`, is created to replace any occurrence of **more than one space** with a **single space** [[07:27](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=447)].
    * The RegEx pattern used is `\s+` (where `\s` is a space character and `+` means "one or more times") [[08:27](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=507)].



## 3. In-Depth NLP Preprocessing with spaCy

With the text cleaned of symbols and excess spaces, the next phase involves standard NLP techniques to prepare the words for deep learning [[10:10](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=610)]. The video utilizes the **spaCy** library, which is known for its efficiency.

### Lemmatization [[11:11](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=671)]

* **Concept:** Lemmatization is the process of converting a word to its base or dictionary form, known as the **lemma**. This is crucial because it groups together different inflected forms of a word (e.g., "running," "runs," "ran" become "run"), reducing the total vocabulary size and improving model generalization.
* **Implementation:** The spaCy library's English module (`en_core_web_sm`) is loaded [[10:54](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=654)]. A function is created to:
    1.  Process the text using the spaCy NLP object.
    2.  Extract the `.lemma_` (lemma) of every word (`token`) in the document.
    3.  Join the lemmas back into a single string for a new column, `lemma_tweet` [[13:36](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=816)].

### Stop Word Removal [[15:38](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=938)]

* **Concept:** **Stop words** are common words that do not carry significant meaning for classification (e.g., "a," "the," "is," "and"). Removing them reduces the feature space and helps the model focus on critical terms.
* **Implementation:** A function iterates through the lemmatized tweets and uses spaCy's built-in `word.is_stop` attribute to check if a word is a stop word. Only words that are **not** stop words are kept and stored in the final column, **`final_tweet`** [[17:28](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1048)].


## 4. Feature Preparation for Deep Learning

Deep learning models cannot process text strings directly; they require numerical input.

### One-Hot Representation (Tokenization) [[19:51](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1191)]

* **Concept:** This process converts each word in the cleaned text into a unique integer identifier (token). It maps every word in the vocabulary to a numerical index.
* **Implementation:** The **`OneHot`** function from Keras is used, along with a specified **`vocabulary_size`** (set to 10,000) [[20:13](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1213)]. This creates a list of lists, where each inner list represents a tweet converted into a sequence of numbers.

### Padding Sequences [[22:38](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1358)]

* **Concept:** The number of words (and thus the numerical length) of each tweet is different [[23:37](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1417)]. Deep learning models, however, require all input sequences in a batch to have the **exact same length**. Padding solves this by:
    * Defining a fixed **`sentence_length`** (set to 20) [[24:19](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1459)].
    * Adding zeros to the shorter sequences until they reach the `sentence_length`.
    * **Pre-padding** is chosen, meaning the zeros are added to the beginning of the sequence [[26:37](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1597)].
* **Implementation:** The `pad_sequences` function from Keras is used to create the final input data, **`embedded_tweet`** [[25:34](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1534)].



## 5. Data Balancing and Splitting

The final pre-modeling steps ensure the data is balanced and partitioned for training.

### Data Imbalance and SMOTE [[28:32](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1712)]

* **Problem:** The class distribution is highly imbalanced, with class '1' (Offensive) having 19,185 samples, but class '0' (Hate Speech) having only 1,430 samples. A model trained on this data would be biased towards the majority class.
* **Solution: SMOTE:** The **Synthetic Minority Over-sampling Technique (SMOTE)** from the `imbalance-learn` library is imported [[29:37](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1777)].
    * SMOTE generates synthetic (artificial) samples for the minority classes (class '0') to equalize their count with the majority class.
    * The **`sampling_strategy`** is set to **`minority`**, focusing only on generating samples for the smallest class [[30:26](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1826)].
    * This process increases the total number of samples from 24,783 to 42,000 [[33:29](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2009)].

### Train-Test Split [[31:15](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=1875)]

The balanced data is split into training and testing sets using the `train_test_split` function from Scikit-learn:

* **Training Set:** 80% of the data (used to train the model).
* **Testing Set:** 20% of the data (used for final, unbiased evaluation).



## 6. Stacked LSTM Model Architecture

The model is a **Stacked LSTM** built using the Keras Sequential API [[33:44](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2024)].


### 1. Embedding Layer [[34:39](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2079)]

This is the first layer and is essential for NLP models. It transforms the integer tokens into dense, fixed-size vectors (embeddings) that capture semantic relationships between words.

* **Input 1 (`vocabulary_size`):** 10,000 (The number of unique tokens).
* **Input 2 (`dimension`):** 50 (The size of the vector representation for each word).
* **Input 3 (`input_length`):** 20 (The length of the padded sequence).

### 2. Stacked LSTM Layers [[35:26](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2126)]

Three LSTM layers are stacked sequentially:

* **Layer 1 (100 Neurons):** `return_sequences=True` is set because the output of this layer needs to be a sequence that is fed into the next LSTM layer.
* **Layer 2 (50 Neurons):** `return_sequences=True` is also set.
* **Layer 3 (50 Neurons):** `return_sequences` is removed (or implicitly set to `False`), as the output of this final layer is a single vector (the final hidden state) to be fed into the classification layer.

### 3. Dense Output Layer [[36:40](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2200)]

This is the final fully connected layer for classification.

* **Neurons:** **3** (corresponding to the three target classes: Hate, Offensive, Neither).
* **Activation Function:** **Softmax**. This function converts the raw scores into a probability distribution, ensuring the output probabilities for the three classes sum to 1.



## 7. Model Training and Final Evaluation

### Model Compilation and Training [[37:15](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2235)]

* **Optimizer:** **Adam** (an efficient optimization algorithm).
* **Loss Function:** **Sparse Categorical Crossentropy**. This is the standard loss function for **multiclass classification** when the target labels (Y) are provided as integers (0, 1, 2) rather than one-hot vectors.
* **Metrics:** **Accuracy**.
* The model is trained over **10 epochs** using a **batch size of 32** [[38:34](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2314)].

### Evaluation Metrics

The final evaluation uses the testing set to assess the model's true performance [[39:21](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2361)].

1.  **Model Accuracy:** The model achieves **89% accuracy** on the unseen testing set [[40:05](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2405)].

2.  **Classification Report [[42:19](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2539)]:** This report details the performance for each class:
    * **Precision:** Out of all instances the model *predicted* as Class X, how many were actually Class X?
    * **Recall:** Out of all actual instances of Class X, how many did the model *correctly* predict?
    * **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure. Due to the use of SMOTE, the F1-scores for Class 0 and 1 are well-balanced at **91%**.

3.  **Confusion Matrix [[42:43](http://www.youtube.com/watch?v=Hrt_gl6r2SI&t=2563)]:** This visual plot shows exactly where the model is making errors:


| Metric | Class 0 (Hate Speech) | Class 1 (Offensive) | Class 2 (Neither) |
| :--- | :--- | :--- | :--- |
| **Correct Classification (Diagonal)** | 90% | 91% | 78% |

The high values on the diagonal (dark blue squares) indicate the percentage of correct classifications for each category, demonstrating that the model is performing efficiently for both hate speech and offensive language detection.


http://googleusercontent.com/youtube_content/0
