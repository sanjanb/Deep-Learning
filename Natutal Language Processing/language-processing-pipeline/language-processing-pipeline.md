# 1. Understanding the NLP Pipeline in spaCy

The NLP Pipeline is the series of processing steps that spaCy applies to raw text to transform it into a structured **`Doc` object** rich with linguistic annotations.

## Blank vs. Loaded Pipeline

| Feature | `spacy.blank('en')` (Blank Pipeline) | `spacy.load('en_core_web_sm')` (Loaded Pipeline) |
| :--- | :--- | :--- |
| **Components** | Contains only the **Tokenizer** by default. | Contains a sequence of **trained components** in a specific order. |
| **Functionality** | Only performs word tokenization and sentence tokenization (if manually added). | Performs advanced NLP tasks like Part-of-Speech tagging, Lemmatization, and NER. |
| **Output** | Basic `Doc` object with tokens and text. | Rich `Doc` object with linguistic properties attached to every token and span. |
| **Purpose** | Used when you need to **customize** the entire pipeline or for basic tasks. | Used for **ready-to-use**, high-performance language analysis. |

The loaded pipelines (`en_core_web_sm`, `en_core_web_md`, etc.) are pre-trained models that contain the necessary components to execute the full pipeline.


# 2. Key Components of a Loaded Pipeline

When you load a pre-trained pipeline, you gain access to a sequence of components that execute one after the other. You can see the names of these components using `nlp.pipe_names`.

The main components discussed are:

### A. Part-of-Speech (POS) Tagger

The tagger assigns a grammatical category to every token in the text. This component gives you the **Part-of-Speech (`token.pos_`)** attribute.

  * **Goal:** To identify if a word is a **Noun**, **Verb**, **Adjective**, **Pronoun**, etc.
  * **Example:** In the sentence "Apple ate a fruit," `Apple` is a Proper Noun (`PROPN`), `ate` is a Verb (`VERB`), and `fruit` is a Noun (`NOUN`).

### B. Lemmatizer

The lemmatizer component is responsible for finding the **base form** or **lemma** of a word.

  * **Goal:** To reduce inflected (or derived) words to their root form. This is crucial for comparing words with the same core meaning.
  * **Example:** The base word (lemma) for the verb "said" is "say." The base word for "ate" is "eat".

### C. Named Entity Recognizer (NER)

The NER component identifies and classifies named entities in the text into pre-defined categories.

  * **Goal:** To automatically detect mentions of real-world objects like people, organizations, locations, money, and dates.
  * **Access:** Entities are stored as **spans** in the `doc.ents` attribute.
  * **Example:** In the phrase "Tesla acquired Twitter for 45 Billion Dollars," the NER component identifies:
      * `Tesla` and `Twitter` as an **ORG** (Organization).
      * `45 Billion Dollars` as **MONEY** (Monetary Value).


# 3. Customizing the Pipeline

spaCy provides flexibility to create custom pipelines, which is useful when you only need a few components or have specialized components you want to add.

### Adding Custom Components

Instead of relying on the full pre-trained pipeline, you can start with a **blank pipeline** and add a specific component from a pre-trained model.

1.  **Start Blank:** Create a blank NLP object: `nlp = spacy.blank('en')`.
2.  **Load Source:** Load the trained pipeline that has the component you want to copy (e.g., `source_nlp = spacy.load('en_core_web_sm')`).
3.  **Add Component:** Use the `add_pipe` method, specifying the source and the component name (e.g., `ner`):
    ```python
    nlp.add_pipe('ner', source=source_nlp)
    ```
4.  **Result:** Your `nlp` object now only contains the **Tokenizer** and the **NER** component, making it faster and more focused on your specific problem.

### Multi-Language Support

spaCy supports many languages. You can download and load pre-trained pipelines for languages other than English, such as French (`fr_core_web_sm`). This allows you to apply the full linguistic processing power (POS tagging, NER, etc.) to text in multiple different languages, making it a versatile tool for global NLP projects.

The video URL is: [http://www.youtube.com/watch?v=hKK59rfpXL0](http://www.youtube.com/watch?v=hKK59rfpXL0)

http://googleusercontent.com/youtube_content/3

http://googleusercontent.com/youtube_content/6
