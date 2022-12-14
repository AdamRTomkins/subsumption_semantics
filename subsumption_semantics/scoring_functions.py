from typing import List
from math import isclose
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from spacy.tokens.doc import Doc
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def token_similarity_scoring(
    query:Doc,
    document:Doc,
    links:List=[],
    threshold:float=0.0,
    allow_entities: bool = True,
    allow_chunks: bool = True,
):

    # Pull in tokens with vectors
    query_tokens = [t for t in query if t.has_vector]
    document_tokens = [t for t in document if t.has_vector]

    if len(query_tokens) == 0:
        logger.warning(f"Query has no tokens with a vector. [{query.text[:50]}...]")
        return []

    if len(document_tokens) == 0:
        logger.warning(f"Document has no tokens with a vector. [{query.text[:50]}...]")
        return []

    # Allow single tokens to match to spans in the document.
    if allow_entities:
        document_tokens.extend([e for e in document.ents if e.has_vector])

    # Allow single tokens to whole noun chunks
    if allow_chunks:
        document_tokens.extend(list(document.noun_chunks))

    # find the maximimum cosine similarity
    cs = cosine_similarity(
        np.array([t.vector for t in query_tokens]),
        np.array([t.vector for t in document_tokens]),
    )

    max_index = np.argmax(cs, axis=1)

    scores = [cs[q, d] for q, d in enumerate(max_index)]
    explanation = [
        "same token" if isclose(s, 1, abs_tol=10**-3) else f"{s} semantic similarity"
        for s in scores
    ]

    return [
        r
        for r in list(
            zip(
                [[q] for q in query_tokens],
                scores,
                [[q] for q in [document_tokens[d] for d in max_index]],
                explanation,
            )
        )
        if r[1] > threshold
    ]


def entity_id_scoring(query:Doc, document:Doc, links:List=[], threshold:float=0.0):
    """ Give identical meaning to entities with the same entity id."""

    def exact_similarity(X:List, Y:List):
        """Simple exact match similarity for entity ids (entities are semantically exact)"""
        res = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                res[i, j] = int(x == y)

        return res

    if len(query.ents) == 0 or len(document.ents) == 0:
        return []

    cs = exact_similarity(
        np.array([[t.ent_id] for t in query.ents]),
        np.array([[t.ent_id] for t in document.ents]),
    )
    max_index = np.argmax(cs, axis=1)

    scores = [cs[q, d] for q, d in enumerate(max_index)]

    explanation = [
        f"same concept [{span.ent_id_}]" if s == 1 else "different"
        for span, s in zip(query.ents, scores)
    ]

    return [
        r
        for r in list(
            zip(
                [[t for t in q] for q in query.ents],
                scores,
                [[t for t in q] for q in [document.ents[d] for d in max_index]],
                explanation,
            )
        )
        if r[1] > threshold
    ]


def hierarchical_entity_scoring(query:Doc, document:Doc, links:List=[], entity_data: Dict = {}):
    """ Give hierarchical meaning to entities with overlapping hierarcy."""

    def _hierarchical_similarity(qentity, dentity, enitity_data):
        """Score the """


    def hierarchical_similarity(X:List, Y:List, entity_data:Dict):
        """Simple exact match similarity for entity ids (entities are semantically exact)"""
        res = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                res[i, j] = int(x == y)

        return res

    if len(query.ents) == 0 or len(document.ents) == 0:
        return []

    cs = hierarchical_similarity(
        [t.ent_id_ for t in query.ents],
        [t.ent_id_ for t in document.ents],
        entity_data
    )

    max_index = np.argmax(cs, axis=1)

    scores = [cs[q, d] for q, d in enumerate(max_index)]

    explanation = [
        f"same concept [{span.ent_id_}]" if s == 1 else "different"
        for span, s in zip(query.ents, scores)
    ]

    return [
        r
        for r in list(
            zip(
                [[t for t in q] for q in query.ents],
                scores,
                [[t for t in q] for q in [document.ents[d] for d in max_index]],
                explanation,
            )
        )
        if r[1] > 0
    ]



def entity_syntax_scoring(query, document, links, threshold=0.5, root_threshold=0.5):
    """Boost entity scores that are linked by similar roots."""

    def get_noun_phrase(token, doc):
        for noun_chunk in list(doc.noun_chunks):
            if token in list(noun_chunk):
                return noun_chunk

    new_links = []
    for link in links:
        # Only look at strong connections for relationship boosting
        if link[1] < threshold:
            continue

        qtoken = link[0][0]
        dtoken = link[2][0]

        # Only look at Entities
        if qtoken.ent_id_ == "" or dtoken.ent_id_ == "":
            continue

        try:
            qchunk = get_noun_phrase(qtoken, query)
            dchunk = get_noun_phrase(dtoken, document)
            qhead = qchunk.root.head
            dhead = dchunk.root.head
        except:
            continue

        root_similarity = qhead.similarity(dhead)

        # Only boost close relationships
        if root_similarity > root_threshold and qchunk.root.dep == dchunk.root.dep:
            link = list(link)
            link[0].append(qhead)
            link[2].append(dhead)
            link[1] = link[1] + qhead.similarity(dhead)
            link[-1] = (
                link[-1] + f". Related verbs {qhead} and {dhead} [{dchunk.root.dep_}]."
            )
            link = tuple(link)
            new_links.append(link)

    return new_links
