# 1. The Core Concept: Normalizing Words

**Stemming** and **Lemmatization** are both essential pre-processing steps in NLP. They are used to **reduce inflected or derived words to a common base form**.

The primary goal is to treat different forms of the same word (e.g., "talking," "talk," "talked") as a single token. This process is crucial for improving the performance and accuracy of NLP applications by:

* **Improving Search/Retrieval:** If a user searches for "talking," they expect results containing "talk" and "talked".
* **Enhancing Classification:** In tasks like sentiment analysis, different forms of a word should contribute to the same core meaning (e.g., "eating" vs. "ate").
* **Reducing Vocabulary Size:** Grouping word variations reduces the overall number of unique features a model has to learn, leading to greater efficiency.


# 2. Stemming: The Heuristic Approach

**Stemming** is a rough, rule-based process that chops off prefixes or suffixes from a word to get a base form, often called a **stem**. It is based on **fixed rules** or **heuristics** and **does not rely on linguistic knowledge**.

* **Process:** It applies simple, dumb rules like: "If the word ends in 'ing,' remove 'ing'" (e.g., *talking* → *talk*).
* **Tool Used (NLTK):** Stemming is demonstrated using the **PorterStemmer** from the **NLTK (Natural Language Toolkit)** library.
* **Pros:**
    * It is **faster** than lemmatization because it only performs quick string manipulations based on fixed rules.
* **Cons (The "Dumb" Problem):**
    * **Over-stemming:** The process can sometimes cut off too much of the word, resulting in a **stem that is not a real, valid word**.
    * **Example:** Applying a simple rule to the word *"ability"* might incorrectly result in the stem *"abil"*, which has no meaning. This is why it's considered less sophisticated than lemmatization.


# 3. Lemmatization: The Linguistic Approach

**Lemmatization** is a more sophisticated process that uses the **linguistic knowledge** (grammar and context) of a language to return the word to its true **dictionary form**, known as the **lemma**.

* **Process:** It uses **Part-of-Speech (POS)** information and a **vocabulary lookup table** to ensure the base form returned is a grammatically correct word.
* **Tool Used (spaCy):** **spaCy** focuses on **Lemmatization** because it provides a more accurate and linguistically sound base form, and therefore, it does not include a stemming component. The lemmatization component is part of a pre-trained NLP pipeline.

## Lemmatization in spaCy

In spaCy, the base word (lemma) is accessed via the **`.lemma_`** attribute of a token.

| Word | Lemmatization Output (`.lemma_`) | Reason/Benefit |
| :--- | :--- | :--- |
| `ate` | `eat` | Correctly identifies the base verb (present tense). |
| `ability` | `ability` | Correctly recognizes the word is already a base noun, avoiding the over-stemming error. |
| `better` | `well` | Maps the comparative form to the base adverb/adjective, demonstrating linguistic intelligence. |

The video notes that the attribute **`.lemma`** (without the underscore) returns a **unique hash identifier** for the lemma, which is used internally by spaCy to manage its fixed vocabulary.



# 4. Customizing the Lemmatization Pipeline

The video demonstrates how to **customize** the lemmatization rules in spaCy to handle slang, misspellings, or non-standard terms specific to a domain, which the default model might not recognize.

### Modifying the `AttributeRuler`
Slang words like "bro" (for "brother") are not in the standard vocabulary, so they need custom rules. This is done by modifying the **`AttributeRuler`** component in the pipeline.

1.  **Retrieve the Component:** Get the existing `AttributeRuler` from the loaded pipeline: `ruler = nlp.get_pipe("attribute_ruler")`.
2.  **Define and Add the Custom Rule:** Use the `add` method to specify a custom lemma for a particular token text:
    * The rule maps the tokens `'bro'` and `'bra'` to the lemma `'brother'`.
    * After adding the rule, both "bro" and "bra" will correctly have a `.lemma_` of "brother," successfully adapting the pre-trained model to specific linguistic patterns.

This customization feature shows how to combine the power of a pre-trained model with the flexibility needed for real-world text where language is often informal or domain-specific.
