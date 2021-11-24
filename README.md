# PLSUM
Generating pt-br Wikipedia articles from refference documents from the web.

This is the official repository for the paper __"PLSUM: Generating PT-BR Wikipedia by Summarizing Websites"__, by André Seidel Oliveira¹ and Anna Helena Reali Costa¹, that is going to be presented at ENIAC 2021.
Our work is inspired by [WikiSum](https://arxiv.org/pdf/1801.10198.pdf) (LIU, Peter J. et al., 2018), a similar work for the English language. 

1 - researchers at the Department of Computer Engineering and Digital Systems (PCS) of University of São Paulo (USP)

__The challenge: Generate Brazilian Wikipedia leads from multiple website texts!__

PLSUM has as input (1) a _Title_ and (2) a _set of texts related to the title_ (both in Portuguese), and returns an _original wiki-like summary about the title_.
PLSUM has two stages: The extractive stage will filter the set of related documents on input, returning a limited amound of sentence, while the abstractive stage generates an abstractive (authorial) summary given the title and extracted sentences.
The model was fine-tuned and tested on [_BrWac2Wiki_](https://github.com/aseidelo/BrWac2Wiki), a dataset with records associating a title, multiple documents from the web, and Wikipedia leads (the first section of a Wikipedia article).

## Modules

Bellow a brief description of what you will find on ```src/``` folder:

### extractive_stage:
The extractive_stage filter prominent sentences from the input documents. 
It returns a list of N sentences in order of importance, where N is a hyperparameter.
On ```src/extractive_stage/sparse_models.py``` we implement _TF-IDF_, _Random_, and _Cheating_ as described in the paper.
On ```src/extractive_stage/cluster_embbeding.py``` and ```src/extractive_stage/generate_embeddings.py``` we implement an extractive stage based on sentence embeddings (IN PROGRESS).

### abstractive_stage:
We compare two Transformer encoder-decoders, fine-tuned on [BrWac2Wiki](https://github.com/aseidelo/BrWac2Wiki) dataset for _Multi-document Abstractive Summarization_: 
[PTT5](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab) (CARMO, Diedre et al., 2020) and [Longformer](https://huggingface.co/allenai/led-base-16384) (BELTAGY, Iz; PETERS, Matthew E.; COHAN, Arman., 2020). 

Our fine-tuned checkpoints for both models are on hugging-face:

- PTT5 fine-tune: [_plsum-base-ptt5_](https://huggingface.co/seidel/plsum-base-ptt5)
- Longformer fine-tune: [_plsum-base-led_]() (IN PROGRESS)

### search_tools:
Codes for searching for content related to a title on the web. 
On ```src/search_tools/get_web_urls.py``` we use [googlesearch](https://pypi.org/project/googlesearch-python/) lib for searching the title on Google.
On ```src/search_tools/get_urls_text.py``` we apply [html2text](https://pypi.org/project/html2text/), [nltk](https://www.nltk.org/), and [langdetect](https://pypi.org/project/langdetect/) to scrap and filter texts in Portuguese from the retrieved urls.

### Usage of ```src/app.py```:

Run summary inferences with:

```python app.py -t '[TITLE_1]' ... '[TITLE_N]' -o [OUTPUT_FILE]```

or 

```python app.py -f [FILE_NAME] -o [OUTPUT_FILE]```,
where ```[FILE_NAME]``` is a file with one title per line.

The algorithm will google the list of titles, scrap texts from retrieved urls, and apply the PLSUM summarization framework to each title, printing the predicted summaries and storing them into ```[OUTPUT_FILE]```. Our default extractive stage is _TF-IDF_ and abstractive stage is [_plsum-base-ptt5_](https://huggingface.co/seidel/plsum-base-ptt5).

## Results

We compared 7 different combinations of extractive and abstractive stages on unseen examples from [BrWac2Wiki](https://github.com/aseidelo/BrWac2Wiki). 
TF-IDF + PTT5 with J (number of input tokens) = 512 had

![Results](https://github.com/aseidelo/plsum/docs/results.png)

## Acknowledgements
This research was supported by _Itaú Unibanco S.A._, with the scholarship program of _Programa de Bolsas Itaú_ (PBI), and partially financed by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES), Finance Code 001, and CNPQ (grant 310085/2020-9), Brazil.
Any opinions, findings, and conclusions expressed in this manuscript are those of the authors and do not necessarily reflect the views, official policy or position of the Itaú-Unibanco, CAPES and CNPq.
