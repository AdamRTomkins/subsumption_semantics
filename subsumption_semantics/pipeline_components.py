from typing import Dict

from spacy.language import Language

from subsumption_semantics.similarity_metrics import (
    late_interaction_similarity,
    #neurosymbolic_similarity,
    complex_similarity
)


@Language.factory("late_interaction_similarity", default_config={})
class LateInteractionSimilarity:
    def __init__(self, nlp, name: str):
        self.name = name

    def __call__(self, doc):
        doc.user_hooks["similarity"] = late_interaction_similarity
        doc.user_span_hooks["similarity"] = late_interaction_similarity
        return doc


#@Language.factory("neurosymbolic_similarity", default_config={"data": {}})
#class NeuroSymbolicSimilarity:
#    def __init__(self, nlp, name: str, data: Dict):
#        self.name = name
#        self.data = data
#
#    def __call__(self, doc):
#        doc.user_hooks["similarity"] = neurosymbolic_similarity
#        doc.user_span_hooks["similarity"] = neurosymbolic_similarity
#        return doc


@Language.factory("complex_similarity", default_config={})
class ComplexSimilarity:
    def __init__(self, nlp, name: str):
        self.name = name

    def __call__(self, doc):
        doc.user_hooks["similarity"] = complex_similarity
        doc.user_span_hooks["similarity"] = complex_similarity
        return doc
