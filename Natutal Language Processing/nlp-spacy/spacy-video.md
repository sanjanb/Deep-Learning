# 1. The Purpose of Tokenization

Tokenization is the process of splitting a continuous piece of text (like a paragraph or a document) into **meaningful smaller units** called **tokens**.

The video discusses two main types [[01:15](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=75)]:

1.  **Sentence Tokenization:** Splitting a paragraph into individual sentences.
2.  **Word Tokenization:** Splitting a sentence into individual words, punctuation marks, and other meaningful segments.

## Why Simple Splitting Fails

The core problem is that simple string operations (like splitting by a space or a period) are **too naive** for human language [[01:35](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=95)].

  * **Example:** In the sentence, "Dr. Strange loves Pav Bhaji.", splitting by the period (`.`) would incorrectly treat "Dr." as a complete sentence and separate the abbreviation, leading to a loss of meaning [[01:50](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=110)].
  * **The Need:** A tokenizer needs **language understanding** and defined **special rules** to correctly identify abbreviations (like "Dr." or "N.Y.") and correctly separate punctuation and currency from words [[02:19](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=139)].



# 2. The spaCy NLP Pipeline and Tokenization

In spaCy, the entire NLP process is handled by a **Pipeline**. When you load a language, you create an `nlp` object that contains all the processing components.

## The Default Blank Pipeline

1.  **Creating the `nlp` Object:** You start by creating a language object, typically using `spacy.blank('en')` for English, which essentially creates a "blank" pipeline with only the **Tokenizer** component active [[03:34](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=214)].
2.  **Creating the `Doc` Object:** When you feed raw text into the `nlp` object (`doc = nlp(text)`), the **Tokenizer** is immediately executed [[04:58](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=298)].
3.  **Output:** The tokenizer takes the raw text and converts it into a **`Doc` object**, which is a container for all the processed tokens. You can then iterate over these tokens: `for token in doc:` [[06:15](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=375)].

## The Three-Stage Tokenization Rules (Linguistically-Informed)

spaCy's tokenizer uses a sophisticated, rule-based approach to ensure accurate segmentation [[09:46](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=586)]. It works in three phases to split words correctly:

1.  **Prefixes:** Separating characters at the beginning of a word (e.g., currency symbols `$` or opening quotation marks `"`).
2.  **Suffixes:** Separating characters at the end of a word (e.g., punctuation marks `!`, `?`, or closing quotation marks `"`).
3.  **Infixes/Exceptions:** Special rules for handling internal characters, like apostrophes (`'s`), hyphens (`-`), and common abbreviations that should remain intact (or be split according to linguistic rules) [[10:30](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=630)].



# 3. Token Attributes and Metadata (The Power of spaCy)

Once the text is converted into a `Doc` object containing individual **Token objects**, the real power of spaCy is revealed. Each token object is rich with linguistic metadata that you can access instantly.

### Core Token Access

  * **Indexing:** Tokens can be accessed just like elements in a Python list (e.g., `doc[0]`, `doc[1]`) [[08:48](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=528)].
  * **Token Text:** The raw text of the token is accessed via the `.text` attribute (e.g., `token.text`) [[16:56](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1016)].

### Token Attributes (Boolean Flags)

Each token has numerous boolean attributes (`is_...`) that provide quick information about its nature without writing complex code. These are extremely useful for filtering and analysis [[15:22](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=922)]:

| Attribute | Description | Example |
| :--- | :--- | :--- |
| **`.is_alpha`** | Is the token purely alphabetical (a word)? | `True` for "Tony" [[16:16](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=976)]. |
| **`.is_digit`** | Is the token a digit (number)? | `False` for "Tony" [[16:38](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=998)]. |
| **`.is_currency`** | Is the token a currency symbol? | `True` for `$` [[17:45](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1065)]. |
| **`.is_punct`** | Is the token punctuation? | `True` for `.` or `!` [[18:07](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1087)]. |
| **`.like_num`** | Does the token look like a number (e.g., "two", or "2")? | `True` for "2" [[17:07](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1027)]. |
| **`.like_email`** | Does the token look like an email address? | `True` for "peter@email.com" [[21:43](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1303)]. |

### Practical Use Case: Email Extraction

The video demonstrates a powerful, real-world application of these attributes: **extracting all email addresses from a text file** [[19:39](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1179)].

Instead of using complex Regular Expressions (regex), which can be tedious, you simply iterate through all tokens and check the boolean flag:

```python
emails = []
for token in doc:
    if token.like_email:
        emails.append(token.text)
```

-----

# ⚙️ 4. Customization and Handling Sentences

### Tokenizer Customization

While spaCy's defaults are great, you can customize the tokenizer to handle specific slang or domain-specific terms that may not be in its default rules (e.g., splitting "gimme" into "gim" and "me") [[27:16](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1636)].

You can modify the `nlp.tokenizer.add_special_case()` method to define how specific strings should be split into multiple tokens.

### Adding the Sentence Tokenizer

Unlike the word tokenizer, the **sentenceizer** component is often **not included** in a blank pipeline. This means if you try to use `doc.sents` immediately after creating a blank `nlp` object, you'll get an error [[30:24](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1824)].

To fix this, you must **manually add the sentenceizer component** to the pipeline:

```python
# Fix the error:
nlp.add_pipe("sentencizer")

# Now this will work:
for sentence in doc.sents:
    print(sentence)
```

This action adds the necessary component so the `Doc` object gains the ability to identify sentence boundaries correctly [[31:02](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=1862)].

-----

# Exercise Summary

The video concludes with an exercise that encapsulates the concepts learned, challenging the user to:

1.  **Extract all URLs** from a provided paragraph of text.
2.  **Extract all transaction money amounts** (e.g., $200 and 500 Euros) using the `token.is_currency` and potentially `token.like_num` attributes [[34:14](http://www.youtube.com/watch?v=_lR3RjvYvF4&t=2054)].

http://googleusercontent.com/youtube_content/2
