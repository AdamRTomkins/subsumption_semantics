import numpy as np
from typing import List, Callable


from sklearn.metrics.pairwise import cosine_similarity

from spacy.tokens.doc import Doc

from subsumption_semantics.scoring_functions import (
    entity_syntax_scoring,
    entity_id_scoring,
    token_similarity_scoring,
)


def late_interaction_similarity(query: Doc, document: Doc):
    """Basic Late Interaction per-token scoring"""

    # Take a QxD token cosine-similarity
    cs = cosine_similarity(
        np.array([t.vector for t in query if t.has_vector]),
        np.array([t.vector for t in document if t.has_vector]),
    )

    # Take the maximum pairwise cosine-similarity between Query and Document
    max_index = np.argmax(cs, axis=1)

    # Do a normalized average token score
    score = sum([cs[q, d] for q, d in enumerate(max_index)]) / len(max_index)

    # Create a basic Explanation of why this score is produced
    token_explain = [(q, d, cs[q, d]) for q, d in enumerate(max_index)]

    return score, token_explain


def information_weighed_late_interaction(query:Doc, document:Doc):
    """Do an information-weight filtered late interaction"""

    # Filter out tokens with limited information content to not muddy the composed vector
    query_tokens = [
        t
        for t in query
        if t._.information_content is None or t._.information_content > 0
    ]
    document_tokens = [
        t
        for t in document
        if t._.information_content is None or t._.information_content > 0
    ]

    # Take a filtered QxD token cosine-similarity
    cs = cosine_similarity(
        np.array([t.vector for t in query_tokens if t.has_vector]),
        np.array([t.vector for t in document_tokens if t.has_vector]),
    )

    # Take the maximum pairwise cosine-similarity between Query and Document
    max_index = np.argmax(cs, axis=1)

    # Do a normalized average token score
    score = sum([cs[q, d] for q, d in enumerate(max_index)]) / len(max_index)

    # Create a basic Explanation of why this score is produced
    token_explain = [(q, d, cs[q, d]) for q, d in enumerate(max_index)]

    return score, token_explain




def complex_similarity(query: Doc, document: Doc, elements: List[Callable] = None):

    if not elements:
        # The basic elements contain token-similarity, entity-id scoring and basic syntax
        elements = [
            token_similarity_scoring,  # standard token-based late-interaction
            lambda x, y, z: entity_id_scoring(
                x, y, z, threshold=0.0
            ),  # entity identity scoring
            entity_syntax_scoring,  # check entities are related in a similar clause structure
        ]

    # Create Candidate Scoring Links with elements above
    links = []
    for element in elements:
        links.extend(element(query, document, links))

    # Link Discrimination
    ## Sort th links by their score, and the number of implicated tokens in the query
    links = sorted(links, key=lambda x: (x[1], len(x[0])), reverse=True)

    ## Iterate over the sorted links, filtering links on already-seen tokens, adding unseen token-based links.
    chosen = []
    top_links = []
    for l in links:
        # Only consider links that are not already covered by a higher score (this will only bring down the average)
        if len(set(l[0] + l[2]).intersection(set(chosen))) == 0:
            chosen.extend(l[0])
            # Lets not re-link with tokens in the document too
            chosen.extend(l[2])
            # Keep this link for the final score.
            top_links.append(l)

    # If we have no links, we have no similarity.
    if len(top_links) == 0:
        return 0, []

    # Link Scoring
    score = sum([l[1] / len(top_links) for l in top_links])

    return score, top_links
