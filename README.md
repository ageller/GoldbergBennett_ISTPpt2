# GoldbergBennett_ISTPpt2
NLP + GAI analysis of Bennett Goldberg's survey data, using [latent-scope](https://github.com/enjalot/latent-scope).


## Relevant links

https://enjalot.substack.com/p/introducing-latent-scope

https://github.com/enjalot/latent-scope

https://pypi.org/project/latentscope/

https://umap-learn.readthedocs.io/en/latest/index.html

https://hdbscan.readthedocs.io/en/latest/index.html

https://github.com/jina-ai/jina

## Installation

(latentscope won't install in python 3.12)

```
conda create --name latentscope-wsl python=3.11 openpyxl
conda activate latentscope-wsl
pip install latentscope
```

## Updates to the latentscope code

I will include a .tgz file with the updated code.  Below is a description of the updates I made.

- Error when trying to label the clusters using any transformer, 'pipeline' not defined.  Looks like the developer forgot to import it... I edded `latentscope/models/providers/transformers.py` to add a line 56 : `from transformers import pipeline`, dito for `torch` .
- updated the prompt in `scripts/label_clusters.py` to choose the `label_length` as an input and allow the LLM more tokens for the label.
- sending `max_new_tokens` as an arg to `chat` in `models/providers/transformers.py`.  This will be from `5*label_length` in `scripts/label_clusters.py` (to allow for some extra space, since the LLM doesn't often respect the label_length specified in the prompt)
- replaced `context` with `instructions` in `scripts/label_clusters.py` to allow the user more freedom for how to define the prompt to the LLM
- added a more careful check for token length when defining the list of items to send to the LLM in `scripts/label_clusters.py` so that it doesn't cut an answer off mid-way.  
- trying to improve the cleanup of the LLM-produced labels in `scripts/label_clusters.py`.
- added an input for `n_components` in the `scripts/umapper.py` script, which required a few edits to `scripts/umapper.py` and also `scripts/cluster.py`


