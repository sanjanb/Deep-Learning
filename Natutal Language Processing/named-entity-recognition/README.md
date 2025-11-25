## 1. What is Named Entity Recognition (NER)?

**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into pre-defined categories such as person, organization, location, expressions of times, quantities, monetary values, and more [[02:05](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=125)].

In essence, NER systems look at a piece of text and automatically tell you what kind of entity each key word or phrase represents. This is more sophisticated than a simple keyword search, as it uses the surrounding context to disambiguate terms (e.g., distinguishing between "Tesla" the company and "Tesla" the person, Nikola Tesla) [[01:38](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=98)].

The entities can be anything from:
* **Person:** *Elon Musk*
* **Organization (ORG):** *Tesla, Google, Twitter*
* **Location (LOC/GPE):** *New York, India, Hong Kong*
* **Monetary Value (MONEY):** *45 billion dollars*
* **Product (PRODUCT):** *Pixel 7* [[02:24](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=144)], [[02:52](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=172)]



## 2. Real-Life Use Cases for NER

The video highlights three primary use cases where NER is invaluable:

### A. Search 
On a news aggregator like Google News or a financial website, NER automatically tags articles with the entities mentioned (e.g., company, product, people). This allows users to search for all news related to a specific **entity** rather than just a keyword [[00:24](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=24)]. The system processes the incoming text and extracts entities like **"Tesla"** (ORG) or **"Pixel 7"** (PRODUCT) to index the article accurately.

### B. Recommendation Systems 
Recommendation engines use NER to understand a user's preferences based on the content they consume.
* **News:** If a user reads articles where the main entities are *Elon Musk* and *Hong Kong*, the system learns the user prefers articles about that person and location, and recommends similar content [[03:45](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=225)].
* **Movies/Content:** For a movie streaming service, the entities are the **Production House** (e.g., Marvel, Pixar, National Geographic) and the **Actors**. By extracting these entities, the system can recommend documentaries or movies from preferred categories or actors [[04:17](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=257)].

### C. Customer Care Service and Routing 
In a customer support system, a user might submit a text-based query. If the system can automatically extract the **course name** (e.g., *Power BI* or *Python*) or **product name** from the complaint as an entity, it can immediately **route the issue** to the correct, specialized support team [[05:06](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=306)]. This saves time and ensures the query is handled by the most qualified agent.



## 3. NER Implementation using spaCy

The video demonstrates how to use the pre-trained NER component within the spaCy NLP pipeline.

### A. Accessing Entities
The `doc` object created by spaCy contains a property called **`doc.ents`** (short for entities), which is a list of all identified entities in the text [[07:29](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=449)].

When iterating through these entities, you can extract:
* **`ent.text`**: The actual word or phrase recognized as the entity (e.g., "Tesla Inc.") [[07:41](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=461)].
* **`ent.label_`**: The classification or label of the entity (e.g., "ORG," "MONEY," "DATE") [[07:52](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=472)].
* **`spacy.explain(label)`**: A helper function to print the full meaning of the label (e.g., "ORG" means "Companies, agencies, institutions, etc.") [[08:14](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=494)].

### B. Visualizing Entities
spaCy provides a powerful tool called **`displacy`** to visually render the entities directly in the output (e.g., in a Jupyter Notebook) [[08:39](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=519)].

* `displacy.render(doc, style="ent")` highlights each entity in the text with a different color and displays its label above it, making it easy to inspect the NER model's performance.

### C. Limitations and Custom Entities (Span)
The video highlights that out-of-the-box NER is **not perfect** and may miss entities or misclassify them (e.g., missing "Twitter" until it's capitalized, or misclassifying a company name) [[09:44](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=584)].

To fix this, the concept of a **Span** is introduced:
* A **Span** is a slice of tokens from a document (like a substring) [[13:50](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=830)].
* You can manually define a specific slice of text as a **Span** object and assign it a custom entity label (e.g., defining tokens 5 to 6 in the document as "Twitter" with the label "ORG") [[14:50](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=890)].
* The **`doc.set_ents()`** function is then used to manually set these custom entities on the document, overriding or adding to the existing ones [[15:44](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=944)].


## 4. Approaches to Building a Custom NER System

Since pre-trained models may have limitations, the video outlines three approaches for building or enhancing an NER system for specific domains (e.g., pharmaceutical data for drug entities):

### A. Simple Lookup (Database-Driven) 
This is the most naive and non-NLP approach, but it works for certain use cases. It involves maintaining a **hard-coded database** or dictionary of entities (companies, locations, drugs). The system simply checks if a token exists in the database and assigns the corresponding label. This requires manual maintenance of the database [[17:54](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1074)].

### B. Rule-Based NER (Entity Ruler) 
This approach uses linguistic rules or patterns to identify entities. spaCy provides the **`EntityRuler`** class for this [[19:42](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1182)].
* **Example Rule:** A proper noun followed by "was born in" and a date is a **Person** entity [[18:54](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1134)].
* **Pattern Matching:** Regular expressions (**Regex**) can be used to define patterns for complex entities like phone numbers or specific codes [[19:31](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1171)].

### C. Machine Learning Approaches 
For advanced, scalable, and context-aware NER, machine learning is necessary. This involves training a model to understand the context of words.
* **Conditional Random Fields (CRF):** A classical machine learning technique used for sequence labeling problems like NER [[20:57](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1257)].
* **BERT (Bidirectional Encoder Representations from Transformers):** A powerful modern deep learning model that excels at understanding context and is now the state-of-the-art for NER [[21:12](http://www.youtube.com/watch?v=2XUhKpH0p4M&t=1272)].



http://googleusercontent.com/youtube_content/1
