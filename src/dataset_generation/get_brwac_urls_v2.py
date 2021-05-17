import json
import lxml.etree as ET
from codecs import open
import argparse
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize 
from hash_tools import HashTable
from filter_urls import check_url

def read_brwac_docs(buffer):
    #print(buffer)
    last_doc_close_pos = buffer.rindex('</doc>')
    buffer_out = buffer[last_doc_close_pos + 6:]
    xml = '<root> ' + buffer[:last_doc_close_pos + 6] + ' </root>'
    parser = ET.XMLParser(encoding="utf-8", recover='True')
    tree = ET.fromstring(xml.encode('utf-8'), parser=parser)
    docs = tree.findall('doc')
    docs_list = []
    unique_words = {}
    i = 0
    for doc in docs:
        s = []
        p = doc.findall('p')
        for para in p:
            new_sent = ''
            for sent in para.findall('s'):
                new_sent = new_sent + sent.text.lower()
            s.append(new_sent)
        sentences = []
        for sentence in s:
            words = sentence.split('\n')
            for word in words:
                if word in unique_words and i not in unique_words[word]:
                    unique_words[word].append(i)
                else:
                    unique_words[word] = [i]
            sentences.append(sentence.replace('\n', ' '))
        url = None
        try:
            url = doc.attrib['uri']
        except:
            pass
        title = None
        try:
            title = doc.attrib['title']
        except:
            pass
        new_dict = {'url' : url, 'title' : title, 'text' : sentences}
        #print(new_dict)
        docs_list.append(new_dict)
        i = i + 1
    #print(unique_words)
    return docs_list, unique_words, buffer_out

def search_on_docs(text, words_dict):
    text_words = text.lower().split(' ')
    inds = None
    for word in text_words:
        try:
            new_inds = words_dict[word]
            #print(word)
            if(inds == None):
                inds = new_inds.copy()
            else:
                inds = [value for value in new_inds if value in inds]
        except:
            if(inds == None):
                inds = []
    #if(len(inds) > 0):
    #    #print('achou')
    #    print(text)
    return inds

def search_on_brwac(wiki_ids, wiki_titles, brwac_file_path):
    hash_table = HashTable(200)
    wiki_urls = []
    for wiki_id in wiki_ids:
        wiki_urls.append({'id' : wiki_id, 'urls' : []})
    buffer_size = 200000000
    total_size = int(22000000000/buffer_size)
    with open(brwac_file_path, 'r', encoding="utf-8") as file:
        buffer = file.read(buffer_size)
        i = 0
        while(len(buffer) > 5):
            docs, unique_words, buffer = read_brwac_docs(buffer)
            if(len(docs) > 0):
                for j in range(len(wiki_ids)):
                    if(len(wiki_urls[j]['urls']) <= 50):
                        docs_inds = search_on_docs(wiki_titles[j], unique_words)
                        for ind in docs_inds:
                            doc = docs[ind]
                            #check if field exists
                            if('url' in doc):
                                # check if uri is one to filter (known offensive websites or wikipedia)
                                if(check_url(doc['url']) is not True):
                                    #wiki_urls[j]['urls'].append(doc.attrib['uri'])
                                    hash_table.set_val(doc['url'], doc['text'])
                                    #print(ss)
                                    wiki_urls[j]['urls'].append(doc['url'])
                                    #wiki_urls[j]['texts'].append(doc.text)
            buffer = buffer + file.read(buffer_size)
            print('{}/{} - buffer size: {}'.format(i, total_size, len(buffer)))
            i = i + 1
            #if(i==5):
            #    break
    return wiki_urls, hash_table

def main(args):
    wiki_titles = []
    wiki_ids = []
    with open(args.wiki_path + args.wiki_file, 'r', encoding="utf-8") as file:
        for line in file:
            content = json.loads(line)
            wiki_titles.append(content['title'])
            wiki_ids.append(content['id'])
    brwac_wiki_urls_dicts, hash_table = search_on_brwac(wiki_ids, wiki_titles, args.brwac_file)
    with open(args.wiki_urls_output_path + args.wiki_file, 'wb') as out_file:
        for wiki in brwac_wiki_urls_dicts:
            out_file.write('{}\n'.format(json.dumps(wiki, ensure_ascii=False)).encode('utf-8'))
    serialized_urls = []
    try:
        with open("{}serialized_urls_list.txt".format(args.urls_sentences_output_path), 'r') as file:
            for line in file:
                serialized_urls.append(line.replace('\n', ''))
    except:
        pass
    new_urls = []
    for i in range(len(hash_table.hash_table)):
        with open("{}{:03d}.json".format(args.urls_sentences_output_path, i), 'ab+') as out_file:
            for url, sentences in hash_table.hash_table[i]:
                if(url not in serialized_urls):
                    new_urls.append(url)
                    url_dict = {'url' : url, 'sentences' : sentences}
                    out_file.write('{}\n'.format(json.dumps(url_dict, ensure_ascii=False)).encode('utf-8'))
    with open("{}serialized_urls_list.txt".format(args.urls_sentences_output_path), 'a+') as file:
        for url in new_urls:
            file.write('{}\n'.format(url))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Wikisum-pt dataset from Wikipedia articles and BrWac corpus.')
	parser.add_argument('--brwac_file', default='data/brwac/brwac-dec13.vert', type=str)
	parser.add_argument('--wiki_path', default='data/wikipedia_articles_json/', type=str)
	parser.add_argument('--wiki_file', default='AA/processed_wiki_00.json', type=str)
	parser.add_argument('--wiki_urls_output_path', default='data/wikipedia_ref_urls_brwac_v2/', type=str)
	parser.add_argument('--urls_sentences_output_path', default='data/brwac_ref_urls_sentences_v2/', type=str)
	args = parser.parse_args()
	# turn-on the worker thread
	#for i in range(args.workers):
	#	threading.Thread(target=worker, daemon=True).start()
	main(args)
