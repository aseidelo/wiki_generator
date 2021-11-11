import argparse
from search_tools.get_web_urls import search_urls
from search_tools.get_urls_text import get_page_text 
from extractive_stage.sparse_models import split_sentences, tfidf
from abstractive_stage.ptt5 import load_ptt5, run_batch

'''
Generates a portuguese wiki-like summary for a title:
- Search for the title on google
- Extract texts for retrieved urls
- Extracts most relevant sentences
- Apply abstractive summarizer
'''

def load_abstractive_model(path, model_name, checkpoint):
    load_ptt5(path, model_name, checkpoint)

def abstractive_infer(docs, titles, batch_size=6):
    return run_batch(docs, titles, batch_size=batch_size)

def extractive_infer(docs, query, n_tokens=None, n_documents=None):
    return tfidf(docs, query, n_tokens=n_tokens, n_documents=n_documents)

def run(args):
    ids_titles = None
    titles = args.title
    # read titles from args
    if(args.filein == None):
        ids_titles = [[i, args.title[i]] for i in range(len(args.title))] # list of lists [[id, title], ...]
    # read titles from file
    else:
        with open(args.filein, 'r') as f:
            ids_titles = [[i, line.replace('\n', '')] for i, line in enumerate(f)]
        titles = [row[1] for row in ids_titles]
    print(ids_titles)
    # load abstractive model: tokenizer, model 
    load_abstractive_model(args.abstractive_path, args.abstractive_model_name, args.abstractive_checkpoint)
    # search url on google n -> number of urls retrieved
    titles_urls = search_urls(ids_titles, n = 8) 
    #print(titles_urls)
    extractive_outputs = []
    for title_info in titles_urls:
        title = title_info[1]
        title_urls = title_info[2]
        urls_sentences = []
        for url in title_urls:
            if(url != ''):
                # retrieve text from urls
                paragraphs = get_page_text(url)
                # split paragraphs in sentences of max 100 tokens
                sentences = split_sentences(paragraphs, 100)
                urls_sentences = urls_sentences + sentences
        #print(urls_sentences)
        # apply extractive stage
        relevant_sentences = extractive_infer(urls_sentences, title, n_tokens=500, n_documents=None)
        #print(relevant_sentences)
        title_info.append(relevant_sentences)
        # formating extractive stage output as the abstractive stage receives it
        extractive_output = title + " </s> "
        for sentence in relevant_sentences:
            extractive_output = extractive_output + sentence + " </s> "
        #print(extractive_output)
        title_info.append(extractive_output)
        extractive_outputs.append(extractive_output)
    # apply abstractive stage on batched titles
    return abstractive_infer(extractive_outputs, titles, batch_size=args.abstractive_batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a portuguese wiki-like summary for a title.')
    parser.add_argument('-t', '--title', nargs='+', type=str)
    parser.add_argument('-f', '--filein', default=None, type=str)
    parser.add_argument('-o', '--fileout', default='summaries.csv', type=str)
    parser.add_argument('--abstractive_path', default='../models/', type=str)
    parser.add_argument('--abstractive_model_name', default='ptt5-base-portuguese-vocab', type=str)
    parser.add_argument('--abstractive_checkpoint', default='checkpoint-29000', type=str)
    parser.add_argument('--abstractive_batch_size', default=6, type=int)
    args = parser.parse_args()
    topics = run(args)
    topics.to_csv(args.fileout, index=False, encoding='utf-8')
    print(topics)
    for topic in topics:
        print(topic)
        print(topic['title'] + ':')
        print(topic['predicted_summary'])