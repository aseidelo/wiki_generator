import json
import lxml.etree as ET
from codecs import open
import argparse
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize 
from hash_tools import HashTable

def read_brwac_docs(buffer):
    #print(buffer)
    last_doc_close_pos = buffer.rindex('</doc>')
    buffer_out = buffer[last_doc_close_pos + 6:]
    xml = '<root> ' + buffer[:last_doc_close_pos + 6].replace('<g/>', '').replace('\n', ' ').replace('<p>', '').replace('</p>', '\n') + ' </root>'
    parser = ET.XMLParser(encoding="utf-8", recover='True')
    tree = ET.fromstring(xml.encode('utf-8'), parser=parser)
    docs = tree.findall('doc')
    return docs, buffer_out

def search_on_doc(text, web_title, web_text):
    try:
        lower_text = text.lower()
        lower_title = web_title.lower()
        lower_web_text = web_text.lower()
        if lower_text in lower_title or lower_text in lower_web_text:
            #print(lower_text)
            #print(lower_title)
            #if('anno domini' in lower_text):
            #    print(lower_text)
            return True
        return False
    except:
        return False

def search_on_brwac(wiki_ids, wiki_titles, brwac_file_path):
    hash_table = HashTable(200)
    wiki_urls = []
    for wiki_id in wiki_ids:
        wiki_urls.append({'id' : wiki_id, 'urls' : []})
    buffer_size = 100000000
    total_size = int(22000000000/buffer_size)
    with open(brwac_file_path, 'r', encoding="utf-8") as file:
        buffer = file.read(buffer_size)
        i = 0
        while(len(buffer) > 5):
            docs, buffer = read_brwac_docs(buffer)
            if(docs is not None):
                for doc in docs:
                    for j in range(len(wiki_ids)):
                        sentences = doc.findall('s')
                        full_text = ''
                        ss = []
                        for sentence in sentences:
                            ss.append(sentence.text)
                            full_text = full_text + sentence.text
                        if(search_on_doc(wiki_titles[j], doc.attrib['title'], full_text)):
                            if('uri' in doc.attrib):
                                if('wikipedia' not in doc.attrib['uri']):
                                    if(doc.attrib['uri'] not in wiki_urls[j]['urls']):
                                        #wiki_urls[j]['urls'].append(doc.attrib['uri'])
                                        hash_table.set_val(doc.attrib['uri'], ss)
                                        #print(ss)
                                        wiki_urls[j]['urls'].append(doc.attrib['uri'])
                                        #wiki_urls[j]['texts'].append(doc.text)
            buffer = buffer + file.read(buffer_size)
            print('{}/{} - buffer size: {}'.format(i, total_size, len(buffer)))
            i = i + 1
            #if(i==50):
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
	parser.add_argument('--workers', help='number of threads to perform web searchs in parallel', default=1, type=int)
	parser.add_argument('--batch_size', help='batch of articles between storages', default=10, type=int)
	parser.add_argument('--brwac_file', default='data/brwac/brwac-dec13.vert', type=str)
	parser.add_argument('--wiki_path', default='data/wikipedia_articles_json/', type=str)
	parser.add_argument('--wiki_file', default='AA/processed_wiki_00.json', type=str)
	parser.add_argument('--wiki_urls_output_path', default='data/wikipedia_ref_urls_brwac/', type=str)
	parser.add_argument('--urls_sentences_output_path', default='data/brwac_ref_urls_sentences/', type=str)
	args = parser.parse_args()
	# turn-on the worker thread
	#for i in range(args.workers):
	#	threading.Thread(target=worker, daemon=True).start()
	main(args)
