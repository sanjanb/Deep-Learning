## 1. In-Depth Knowledge of Part of Speech (POS) in English Grammar

Part of Speech (POS) tagging is the process of labeling words in a sentence as corresponding to a particular part of speech, based on both its definition and its context. The video provides a clear overview of the fundamental eight categories:

### Nouns, Verbs, and Pronouns
* **Noun** (N): A person, place, thing, or idea.
    * *Examples:* "Elon," "Mars," "fruits" [[00:33](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=33)].
* **Verb** (V): An action or state of being.
    * *Examples:* "eating" (action), "play" (action) [[00:43](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=43)].
* **Pronoun** (PRON): A word that replaces a noun to avoid repetition; a substitute of a noun.
    * *Examples:* "He," "she," "our," "you," "they," "I" [[01:25](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=85)], [[01:39](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=99)].

### Descriptors: Adjectives and Adverbs
* **Adjective** (ADJ): A word that describes a **noun** or adds more meaning to it.
    * *Examples:* "many" fruits, "sweet" fruits, "red" Tesla, "smart" engineer [[02:06](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=126)], [[02:46](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=166)].
* **Adverb** (ADV): A word that describes a **verb**, an **adjective**, or another **adverb**. It adds meaning to the activity.
    * *Examples:* "slowly" ate, "quickly" ran, "always" scores [[04:48](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=288)], [[05:07](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=307)].

### Connectors and Exclamations
* **Interjection** (INTJ): A word that expresses a sudden, strong emotion or expression.
    * *Examples:* "Wow," "Hey," "Alas" [[06:33](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=393)], [[06:52](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=412)].
* **Conjunction** (CCONJ): A word used to connect different groups of words, phrases, or clauses.
    * *Examples:* "but," "and," "or," "either...or" [[07:29](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=449)], [[07:48](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=468)].
* **Preposition** (ADP/PREP): A word that links a noun to another word in the sentence, often indicating location, direction, or time. They are powerful as they can drastically change the meaning of a sentence.
    * *Examples:* "on" the bus, "at" the bus, "in" the bus [[07:54](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=474)], [[08:36](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=516)].



## 2. POS Tagging in Natural Language Processing (NLP) with spaCy

The video transitions into the practical implementation of POS tagging using **spaCy**, a popular open-source library for advanced NLP.

### Core Implementation Steps
1.  **Loading the Model:** The first step is to import the `spacy` library and load a trained language model (e.g., `en_core_web_sm`) [[10:08](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=608)].
2.  **Creating a Document Object:** The text is processed by the model to create a `doc` object, which contains all the processed linguistic information [[10:29](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=629)].
3.  **Basic POS Tag (`token.pos_`):** This returns the **coarse-grained** part of speech, which generally corresponds to the eight basic categories of English grammar (e.g., `NOUN`, `VERB`, `PROPN`).
4.  **Detailed Tagging (`token.tag_`):** This returns the **fine-grained** tag, which is a further, more detailed sub-categorization of the basic POS.
    * *Example:* The word "made" has a basic POS of `VERB`, but its detailed tag is `VBD` (Verb, past tense) [[15:41](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=941)].
    * *Example:* A proper noun like "Elon" is tagged as `PROPN` (Proper Noun) for basic POS, but `NNP` (Noun, proper singular) for the detailed tag.
5.  **Explaining Tags:** The `spacy.explain(tag_code)` function is crucial for understanding the meaning of the short tag codes (e.g., explaining that `VBD` stands for "verb, past tense") [[12:34](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=754)], [[16:06](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=966)].

### Advanced NLP Insights from spaCy

* **Intelligence in Tagging:** spaCy is smart enough to differentiate between tenses based on the context of the sentence. For instance, it correctly tags "quits" as **third person singular present tense** and "quit" (without 's') as **past tense** [[17:10](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1030)], [[18:18](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1098)].
* **Extended Tag Set:** The actual number of POS tags used in modern NLP libraries like spaCy is much longer than the basic eight, as it includes more specific categories like **Numeral**, **Punctuation**, **Determiner**, and specific forms of **Adposition** (a broader term for prepositions) [[11:46](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=706)], [[14:52](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=892)].



## 3. Real-World Applications of POS Tagging

Knowing the part of speech for every token is extremely useful for building practical NLP applications.

### Application 1: Text Preprocessing and Cleaning
A common initial step in NLP is to clean the input text, which can be done efficiently using POS tags [[19:22](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1162)].
* **Goal:** Filter out tokens that do not contribute to the semantic meaning of the text.
* **Method:** By iterating through the tokens in a `doc` object, you can filter based on `token.pos_`.
    * Tokens commonly filtered out include:
        * `SPACE`
        * `PUNCT` (Punctuation)
        * `X` (Other "garbage" or non-useful characters/tokens) [[21:10](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1270)].
* **Result:** This process cleans the text by removing unnecessary elements, leaving only the meaningful words for subsequent analysis.

### Application 2: Counting Part of Speech Categories
The video demonstrates how to get a quick summary of the document's content by counting the frequency of each POS category.
* **Tool:** The convenient spaCy API `doc.count_by(spacy.attrs.POS)` is used to count tokens grouped by their part of speech [[22:33](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1353)].
* **Usage:** By iterating through the resulting count dictionary and using `spacy.explain()` to translate the numerical POS ID into a readable string, you can see a summary of the text's composition.
* **Insights:** This allows a quick analysis of the text structure, for example, determining if the text is noun-heavy (informational), verb-heavy (action-oriented), or contains a high number of numerals (financial report) [[24:00](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1440)].

---
### **Exercise and Further Learning**
The video concludes with an exercise for the viewer: to apply the concepts learned to extract all **nouns** and **numbers** (numerals) from a story, and then print their respective counts [[24:26](http://www.youtube.com/watch?v=gdHWoQWZGkk&t=1466)].

* **Video URL:** [Part Of Speech POS Tagging: NLP Tutorial For Beginners - S1 E11](http://www.youtube.com/watch?v=gdHWoQWZGkk)



http://googleusercontent.com/youtube_content/0
