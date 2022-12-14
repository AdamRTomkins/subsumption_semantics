import spacy

nlp = spacy.load("en_core_web_md")

## Extend the model with more components

# Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

nlp.add_pipe("late_interaction_similarity")

# List of Entities and Patterns
patterns = [
    {"label": "animal", "pattern": "cat", "id": "0"},
    {"label": "animal", "pattern": "dog", "id": "1"},
    {"label": "animal", "pattern": "feline", "id": "0"},
    {"label": "drug", "pattern": "xabf", "id": "2"},
    {"label": "drug", "pattern": "xabalef", "id": "2"},
    {"label": "drug", "pattern": "drug1", "id": "3"},
    {"label": "drug", "pattern": "drug2", "id": "4"},
    {"label": "symptom", "pattern": "symptom1", "id": "5"},
    {"label": "symptom", "pattern": "symptom2", "id": "6"},
]

ruler.add_patterns(patterns)

# Create your Documents
query = nlp("I have a fast cat on xabf too and a dog called bob")
document = nlp("you have bob on quick feline xabalef")

print("Late Interaction Similarity: ", query.similarity(document))
