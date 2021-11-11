from codecs import open
import json
import argparse
import nltk
from nltk import word_tokenize 
from gensim import corpora, models, similarities
#from gensim.summarization import bm25
import jieba
import re


'''

Sparse extractive techniques

'''

def detect_clone(text, article_sections):
    text_tokens = word_tokenize(text)
    for section in article_sections:
        if(len(section)>0):
            section_tokens = word_tokenize(section.lower())
            #print(list(section_unigrams))
            count_intersection = len(set(section_tokens) & set(text_tokens))
            clone_prob = float(count_intersection)/len(section_tokens)
            #print(count_intersection, len(section_tokens), len(text_tokens), clone_prob)
            #print(clone_prob)
            if(clone_prob > 0.5):
                #print(section, text)
                return True
    return False

def index(docs, query, n_tokens=None, n_documents=None):
    to_out = []
    if(n_tokens != None):
        n = 0
        for doc in docs:
            n = n + len(word_tokenize(doc))
            if(n > n_tokens):
                break
            to_out.append(doc)
    elif(n_documents != None):
        for i in range(n_documents):
            to_out.append(docs[i])
    return to_out

def tfidf(docs, query, n_tokens=None, n_documents=None):
    texts = [filter_paragraph(text).replace('  ', ' ').split(' ') for text in docs]
    #print(texts)
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    #print(word_tokenize(query))
    #print(texts)
    kw_vector = dictionary.doc2bow(query.replace('  ', ' ').split(' '))
    #print(query)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
    scores = index[tfidf[kw_vector]]
    #print(scores)
    to_out_ind = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    #print(to_out_ind)
    to_out = []
    if(n_tokens != None):
        n = 0
        for ind in to_out_ind:
            n = n + len(word_tokenize(docs[ind]))
            if(n > n_tokens):
                break
            to_out.append(docs[ind])
    elif(n_documents != None):
        for ind in to_out_ind[:n_documents]:
            to_out.append(docs[ind])
    return to_out
'''
def bm25(docs, query, n_documents):
    retriever = bm25.BM25(docs)
    scores = retriever.get_scores(query)
    to_out = sorted(range(len(scores)), key=lambda i: scores[i])[-n_documents:]
    return to_out
'''
# recall of bigrams
def cheating(docs, query, n_tokens=None, n_documents=None):
    query_bigrams = set(nltk.bigrams(word_tokenize(query)))
    scores = []
    #print(query_bigrams)
    for doc in docs:
        current_overlap_bigram_count = 0
        doc_bigrams = set(nltk.bigrams(word_tokenize(doc)))
        #print(doc_bigrams)
        intersection = (query_bigrams & doc_bigrams)
        new_score = float(len(intersection))/len(query_bigrams)
        scores.append(new_score)
        #print(new_score)
    to_out_ind = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    to_out = []
    if(n_tokens != None):
        n = 0
        for ind in to_out_ind:
            #print(scores[ind])
            n = n + len(word_tokenize(docs[ind]))
            if(n > n_tokens):
                break
            to_out.append(docs[ind])
    elif(n_documents != None):
        for ind in to_out_ind[:n_documents]:
            to_out.append(docs[ind])
    return to_out

def filter_paragraph(p):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy .
    p = re.sub(r"([?.!,¿()])", r" \1 ", p)
    p = re.sub(r'[" "]+', " ", p)
    # substituir tudo por espaço exceto (a-z, A-Z, ".", "?", "!", ",", letras com acentos da lingua pt)
    p = re.sub(r"[^a-zA-ZçÇéêíáâãõôóúûÉÊÍÁÂÃÕÔÓÚÛ0-9]+", " ", p).lower()
    return p

def split_sentences(docs, max_len):
    to_out_docs = []
    for doc in docs:
        tokens = word_tokenize(doc)
        if len(tokens) > max_len:
            i = 0
            sub_doc = ''
            n_tokens = 0
            for sentence in doc.split('.'):
                n_tokens = n_tokens + len(word_tokenize(sentence))
                if(n_tokens > max_len):
                    to_out_docs.append(sub_doc)
                    sub_doc = ''
                    n_tokens = 0
                    continue
                sub_doc = sub_doc + sentence
        elif (len(tokens) > 5): # at least 5 tokens in 1 sentence
            to_out_docs.append(doc)
    return to_out_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Wikisum-pt dataset from Wikipedia articles and BrWac corpus.')
    parser.add_argument('--dataset_path', default='/home/seidel/Projects/wiki_generator/data/wikisum_ptbr/', type=str)
    parser.add_argument('--predictions_path', default='/home/seidel/Projects/wiki_generator/data/extractive_stage/', type=str)
    parser.add_argument('--input_file', default='input_with_punc.csv', type=str)
    parser.add_argument('--target_file', default='output_with_punc.csv', type=str)
    parser.add_argument('--N', default=1000, type=int)
    args = parser.parse_args()
    with open (args.dataset_path + args.input_file, 'r') as input_file:
        with open (args.dataset_path + args.target_file, 'r') as targets_file:
            with open(args.predictions_path + args.input_file + str(args.N) + '.index', 'wb') as index_file:
                with open(args.predictions_path + args.input_file + str(args.N) + '.tfidf', 'wb') as tfidf_file:
                    with open(args.predictions_path + args.input_file + str(args.N) + '.cheating', 'wb') as cheating_file:
                        i = 0
                        for input_line in input_file:
                            #print(input_line)
                            output_line = targets_file.readline()
                            title_webs = input_line.split('</s>')
                            query = title_webs[0]
                            print(query, i)
                            docs = title_webs[1].split('<\s>')
                            #query = input_content[0]
                            #docs = input_content[1:-1]
                            #print(docs)
                            N = 1000 # number of tokens for output
                            #print('========== Query: ==========\n')
                            #print(query)
                            #print('\n')
                            #print('========== Index: ==========\n')
                            to_out = ''
                            for doc in index(docs, query, n_tokens = N):# , n_documents = 5):
                                to_out = to_out + doc.replace('\n', '') + ' </s>'
                            index_file.write('{}\n'.format(to_out).encode('utf-8'))
                            #print('\n')
                            #print('========== TFIDF: ==========\n')
                            to_out = ''
                            for doc in tfidf(docs, query, n_tokens = N):# , n_documents = 5):
                                to_out = to_out + doc.replace('\n', '') + ' </s>'
                            tfidf_file.write('{}\n'.format(to_out).encode('utf-8'))
                            #print('\n')
                            #print('========== Cheating: ==========\n')
                            to_out = ''
                            for doc in cheating(docs, output_line, n_tokens = N):# , n_documents = 5):
                                to_out = to_out + doc.replace('\n', '') + ' </s>'
                            cheating_file.write('{}\n'.format(to_out).encode('utf-8'))
                            i = i + 1
                            #print('\n')
                            #print('========== Wiki: ==========\n')
                            #print(output_line)
                            #print('\n')
                            #print(bm25(docs, json_content['wiki_title'], 10))
