## Subsumption Semantics

__ A late-interaction approach to semantic similarity for SpaCy, which allows for customization including symbolic integrations. __

## About

With word-embeddings, late-interaction vector composition provides a way to compare documents, without the information bottleneck caused by collapsing documents down to a single vector. 

This work has been inspired by the ColBERT.v2 late-interaction over BERT model, which presents an end-to-end differentiable approach to late-interaction.

In contrast to ColBERT, the Late-interaction approach implemented here is not end-to-end differentiable, but instead allows to modularity in how we compose
our late-interactions, expanding the token-token maximum similarity approach of Colbert, to a generic, configurable set of interaction-processing steps.

These modular steps, allow users to define the modes of document-query interactions with are specifically important to their document similarity use case, for example:
1. Symbolic Interactions
2. Syntactic Structures
3. Entity Interactions
4. Dynamic Semantic Compositionality
5. Custom Vectorisation operations
6. Arbitary vector representations

## Requirements and Installation

For successful similarity metrics, the underlying SpaCy model must include vectors. The default examples are given with the standard Spacy en-core-web-md.
Other models can be installed as required. This repository comes with the ability to install en-core-web-md and en-core-web-lg for Spacy 3.0.0, using 
`pip install -e .[web-md,web-lg]

## Install

```
virtualenv -p python3 env
source env/bin/activate
pip install -e .
```
or Conda:
```
conda create --name spacy python=3.8 -y
pip install -e .
```

Alternatively you can install specific packages like so:
```
pip install -e .[web_lg,api,demo]
```

## Usage

Using the one of the late-interaction similarity metrics as part of a SpaCy model pipeline, all we must do, is add the specific component:

```
import spacy
nlp = spacy.load("en_core_web_md")

## Extend the model with more components
...

nlp.add_pipe("late_interaction_similarity")

query = nlp(...)
document = nlp(...)

similarity = query.similarity(document)

```

## Issues

If you find a bug, please report it on the GitHub issues list. Additionally, if you have feature requests or questions, feel free to post there as well. I'm happy to consider suggestions and Pull Requests to enhance the functionality and usability of the module.

# Developer ToDo:

## Core Representations
1. Improve the Explainability formats, perhaps have spans and relationships and descriptions
2. Create an example of how to visualise the explainability.
## Expanded Scoring Regiems
2. Add a Hierarchical Entity Scorer with "related", "general-bias", "specific-bias", using a core representation of only the parents.
3. Add a generic extra-entity appending mechanism/class of "token" that can represent meta-data key-value-hierarch triples.
4. AMR or Role and Reference Grammary Scoring
