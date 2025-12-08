## 1. The Necessity of N-grams: Overcoming Bag of Words Limitations

The video begins by highlighting a critical flaw in the **Bag of Words (BoW)** model, which treats text as an unordered collection of individual words or tokens (1-grams).

* **The Flaw in BoW:** BoW fails to capture the **order and relationship between words** [[00:23](http://www.youtube.com/watch?v=nZromH6F7R0&t=23)].
* **Importance of Word Order:** In any language, the meaning of a sentence is heavily dependent on the sequence of words. Changing the order can completely alter or destroy the meaning (e.g., "The cat sat on the mat" vs. "The mat sat on the cat") [[00:37](http://www.youtube.com/watch?v=nZromH6F7R0&t=37)].
* **The Solution:** To capture these word relationships and order, the model must consider *sequences* of words instead of just individual words. This is the foundation of the Bag of N-grams model.

***

## 2. Core Concept: Understanding the Bag of N-grams (BoN) Model

The N-gram model is a powerful extension of BoW that solves the word-order problem by creating tokens out of sequences of $N$ consecutive words.

### N-gram Terminology

* **N-gram:** The generic term for a contiguous sequence of $N$ items (in this case, words) from a given sample of text [[01:41](http://www.youtube.com/watch?v=nZromH6F7R0&t=101)].
* **Unigram (1-gram):** A single word. This is equivalent to the traditional **Bag of Words** model [[02:12](http://www.youtube.com/watch?v=nZromH6F7R0&t=132)].
* **Bi-gram (2-gram):** A sequence of two consecutive words (a moving window of two words).
    * *Example:* For the text "devil sat on a sofa," the bi-grams are: "devil sat," "sat on," "on a," "a sofa" [[01:18](http://www.youtube.com/watch?v=nZromH6F7R0&t=78)].
* **Tri-gram (3-gram):** A sequence of three consecutive words (a moving window of three words) [[01:35](http://www.youtube.com/watch?v=nZromH6F7R0&t=95)].
    * *Example:* The tri-grams would be: "devil sat on," "sat on a," "on a sofa."


### How the Vocabulary is Built

The BoN model creates a vocabulary by taking the **union of all unique N-grams** found across all documents in the corpus (the entire text dataset) [[03:35](http://www.youtube.com/watch?v=nZromH6F7R0&t=215)].

* **Vectorization:** Once the N-gram vocabulary is established, each document is converted into a **feature vector**. The value at each index of the vector is the **frequency (count)** of the corresponding N-gram in that document [[04:44](http://www.youtube.com/watch?v=nZromH6F7R0&t=284)].
* **Meaningful Representation:** By using bi-grams and tri-grams, you capture more meaningful phrases (e.g., "devil sat on") compared to individual, disconnected words (e.g., "devil," "sat," "on") [[02:07](http://www.youtube.com/watch?v=nZromH6F7R0&t=127)].
* **Combined Approach:** A common and effective practice is to combine **1-grams and 2-grams** (or even 3-grams) in a single model, as this creates a more meaningful and robust text representation that captures both individual word counts and important word relationships [[05:05](http://www.youtube.com/watch?v=nZromH6F7R0&t=305)].

***

## 3. Practical Drawbacks of the N-gram Model

While effective, N-gram models introduce significant computational and data-related challenges:

* **Increased Dimensionality:** As the value of $N$ (the number of words in the sequence) increases, the size of the vocabulary grows exponentially. This is because the number of possible unique N-gram combinations becomes massive [[06:18](http://www.youtube.com/watch?v=nZromH6F7R0&t=378)].
* **Increased Sparsity:** A large vocabulary leads to a high-dimensional feature vector, where most of the values are **zero**. This happens because any given document only contains a tiny fraction of the total possible N-grams. A sparse matrix is computationally and memory-intensive [[06:26](http://www.youtube.com/watch?v=nZromH6F7R0&t=386)].
* **Out-of-Vocabulary (OOV) Problem:** Like BoW, BoN models do not address the OOV problem. If a model is trained on a set vocabulary, and a new, unseen N-gram appears during prediction, the model cannot represent it, making it difficult to score and classify the new text [[07:05](http://www.youtube.com/watch?v=nZromH6F7R0&t=425)].

***

## 4. Coding Implementation and Case Study

The video demonstrates the implementation using Python's **Scikit-learn** library for a real-world **News Category Classification** problem.

### 4.1. Text Pre-processing (Using spaCy)

Before vectorization, a crucial step is pre-processing to clean the text and improve model performance.

| Pre-processing Step | Purpose | Implementation Details | Timestamp |
| :--- | :--- | :--- | :--- |
| **Stop Word Removal** | Removes common, non-informative words (e.g., "is," "the," "a") that don't add much value to the meaning. | Checks `token.is_stop` property in spaCy. | [[10:31](http://www.youtube.com/watch?v=nZromH6F7R0&t=631)] |
| **Punctuation Removal** | Removes punctuation marks. | Checks `token.is_punct` property in spaCy. | [[10:41](http://www.youtube.com/watch?v=nZromH6F7R0&t=641)] |
| **Lemmatization** | Converts a word to its base or dictionary form (e.g., "eating" $\to$ "eat," "ate" $\to$ "eat"). This helps reduce the vocabulary size and ensures words with the same meaning are treated as one token. | Uses `token.lemma_` property in spaCy. | [[11:10](http://www.youtube.com/watch?v=nZromH6F7R0&t=670)] |

### 4.2. The `CountVectorizer` and `ngram_range`

The video uses Scikit-learn's `CountVectorizer` class for both BoW and BoN implementation [[07:34](http://www.youtube.com/watch?v=nZromH6F7R0&t=454)].

* **`ngram_range` Parameter:** This is the key parameter for N-grams. It takes a tuple `(min_n, max_n)`.
    * **Default `(1, 1)`:** Uses only Unigrams (BoW) [[08:08](http://www.youtube.com/watch?v=nZromH6F7R0&t=488)].
    * **`(2, 2)`:** Uses only Bi-grams [[08:23](http://www.youtube.com/watch?v=nZromH6F7R0&t=503)].
    * **`(1, 2)`:** Uses both Unigrams and Bi-grams (the recommended combined approach) [[08:45](http://www.youtube.com/watch?v=nZromH6F7R0&t=525)].

### 4.3. Data Balancing (Under-Sampling)

The news category dataset used has a significant **class imbalance** (some categories like Business and Sports have far more articles than Science) [[18:53](http://www.youtube.com/watch?v=nZromH6F7R0&t=1133)].

* **Problem:** An imbalanced dataset can cause the model to be biased towards the majority class, leading to poor performance on minority classes.
* **Solution (Under-sampling):** The simplest technique used is under-sampling, where the number of samples in the majority classes is randomly reduced to match the number of samples in the minority class (the minimum count) [[19:55](http://www.youtube.com/watch?v=nZromH6F7R0&t=1195)]. This creates a **balanced dataset** for training.

### 4.4. Model Training and Performance Comparison

A machine learning pipeline is constructed using **`CountVectorizer`** followed by a **Multinomial Naive Bayes** classifier (a model often recommended for text classification problems) [[27:09](http://www.youtube.com/watch?v=nZromH6F7R0&t=1629)].

The training process uses the `stratify` parameter in `train_test_split` to ensure that the training and testing sets maintain the same proportion of classes as the original data [[26:01](http://www.youtube.com/watch?v=nZromH6F7R0&t=1561)].

The key performance comparison is made by training the model with three different vectorizer settings and observing the **classification report** metrics (Precision, Recall, F1-Score):

1.  **BoW (1-gram):** Simple Bag of Words.
2.  **BoN (1-gram and 2-gram):** Combined approach.
3.  **BoN (1-gram, 2-gram, and 3-gram):** Combined approach up to Tri-grams.

| Finding | Conclusion |
| :--- | :--- |
| **N-gram Performance:** | In this specific problem, the simple Bag of Words (1-gram) model performed slightly better than the combined N-gram models (e.g., up to 3-grams) [[30:19](http://www.youtube.com/watch?v=nZromH6F7R0&t=1819)]. **Note:** This highlights that the best technique is always dependent on the specific dataset and requires **trial and error** [[30:49](http://www.youtube.com/watch?v=nZromH6F7R0&t=1849)]. |
| **Impact of Pre-processing:** | By comparing the performance of a model trained on **raw text** versus a model trained on **pre-processed text** (after stop word removal and lemmatization), the model with pre-processed text showed a significant increase in its F1-Score and overall performance [[34:39](http://www.youtube.com/watch?v=nZromH6F7R0&t=2079)]. This emphasizes that **text pre-processing is generally a recommended best practice** in NLP [[35:31](http://www.youtube.com/watch?v=nZromH6F7R0&t=2131)]. |

The video concludes by stressing the importance of **practice** with the provided code exercises, as learning machine learning and NLP is like swimming—you must actively participate and code [[36:14](http://www.youtube.com/watch?v=nZromH6F7R0&t=2174)].

***

You can find the video here: [Text Representation Using Bag Of n-grams: NLP Tutorial For Beginners - S2 E5](http://www.youtube.com/watch?v=nZromH6F7R0)


http://googleusercontent.com/youtube_content/0
