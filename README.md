# wiki_generator
Generating pt-br Wikipedia articles from refference documents from the web.

This is the official repository for the paper "PLSUM: Generating PT-BR Wikipedia by Summarizing Websites", by André Seidel Oliveira¹ and Anna Helena Reali Costa¹, that is going to be presented at ENIAC 2021.
Our work is inspired by [WikiSum] (https://arxiv.org/pdf/1801.10198.pdf), a similar work for the English language. 

1 - researchers at the Department of Computer Engineering and Digital Systems (PCS) of University of São Paulo (USP)

__The challenge: Generate Brazilian Wikipedia articles from multiple website texts!__

PLSUM has as input (1) a _Title_ and (2) a _set of texts related to the title_ (both in Portuguese), and returns an _original wiki-like summary about the title_.
The model has two stages: The extractive stage will filter the input set 
Bellow we show a brief description of each module inside ```src/```:

## extractive_stage:

## abstractive_stage:


## search_tools:
Codes for searching for content related to a title on the web. 
On ```search_tools/get_web_urls.py``` we use [googlesearch] (https://pypi.org/project/googlesearch-python/) lib for searching the title on Google.
On ```search_tools/get_urls_text.py``` we apply [html2text] (https://pypi.org/project/html2text/), [nltk] (https://www.nltk.org/), and [langdetect] (https://pypi.org/project/langdetect/) to scrap and filter texts _in Portuguese_ from the retrieved urls.

### _docids.json_:
Shows the BrWac docs related to each Wikipedia article. Each line is a json entry relating a unique Wikipedia article identifier, _wiki_id_, to several BrWac unique identifiers for documents, _docids_. Each BrWac document cite all the words from the Wikipedia article title, _wiki_title_, at least once. 
Example:
```json
{
  "wiki_id": "415", 
  "wiki_title": "Hino da Independência do Brasil", 
  "docids": ["net-6bb71a", "nete-1e5c7d", "neth-1682c"],
}
```
- _wiki_id_: is the Portuguese Wikipedia _entity id_ for "Hino da Independência do Brasil";
- _wiki_title_: is the title of a Wikipedia article;
- _docids_: is a list of document unique ids from BrWac. Each document is the text content from an website;

### _input.csv_:
Each line has the title for a wiki article and the __sentences__ (document's extracts with a maximum of 100 words) from the BrWac documents associated to the article, separated by the symbol _</s>_. __Lines in the same order as docids.json__.
Example:
```
1  astronomia </s> veja nesta página do site - busca relacionada a astronomico com a seguinte descrição - astronomico </s> astronômico dicionário informal significado de astronômico o que é astronômico substivo masculino referente a corpos celestes como estrelas planetas satélites. </s> (...)
2  (...)
```

### _output.csv_ :
Each line contains the __lead section__ for a Wikipedia article, __also in the same order as docids.json__.
Example:
```
1  O Hino da Independência é uma canção patriótica oficial comemorando a declaração da independência do Brasil, composta em 1822 por Dom Pedro I. A letra foi escrita pelo poeta Evaristo da Veiga.
2  (...)
```

## Details
The search for association between BrWac documents and Wikipedia articles was made with the help of a MongoDB database. We populated the database with BrWac documents and them perform a text search for Wikipedia titles. 

For time reasons, the search had the following rule:
- Search for __every word on the article title__ (AND search);
- Limit a maximum of 15 documents per wiki article;
- Search for __2 seconds__ at least __1 document__, if not found, remove wiki article from dataset.

## Acknowledgements
This research was supported by _Itaú Unibanco S.A._, with the scholarship program of _Programa de Bolsas Itaú_ (PBI), and partially financed by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES), Finance Code 001, and CNPQ (grant 310085/2020-9), Brazil.
Any opinions, findings, and conclusions expressed in this manuscript are those of the authors and do not necessarily reflect the views, official policy or position of the Itaú-Unibanco, CAPES and CNPq.
