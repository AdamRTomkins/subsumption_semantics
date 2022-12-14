import spacy
from subsumption_semantics.similarity_metrics import complex_similarity

# Create a new Model
nlp = spacy.load("en_core_web_md")
## Extend the model with more components
# Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler", name="4")

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


inbuilt = lambda x, y: x.similarity(y)

q = nlp("drug1 reduces symptom2. drug2 masks symptom1")
d1 = nlp("drug1 masks symptom2. drug2 reduces symptom1")
d2 = nlp("drug1 lowers symptom2. drug2 hides symptom1")


print("Document:           ", q)
print("  Shuffled Document:", d1)
print("  Modified Document:", d2, "\n")


print(
    "Cosine Similarity: ",
    "\n  Shuffled",
    round(inbuilt(q, d1), 3),
    "\n  Modified",
    round(inbuilt(q, d2), 3),
    "\n",
)

print(
    "Compositional Similarity: ",
    "\n  Shuffled",
    round(complex_similarity(q, d1)[0] / complex_similarity(q, q)[0], 3),
    "\n  Modified",
    round(complex_similarity(q, d2)[0] / complex_similarity(q, q)[0], 3),
    "\n",
)

print("Self Similarity: ", round(complex_similarity(q, q)[0], 3))
