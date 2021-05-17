import argparse
import json
from os import listdir
from os.path import isfile, join
from hash_tools import HashTable
from codecs import open
import sqlite3
from nltk import word_tokenize 

def detect_clone(text, article_sections):
    text_tokens = word_tokenize(text)
    for section in article_sections:
        if(len(section)>0):
            section_tokens = word_tokenize(section.lower())
            #print(list(section_unigrams))
            count_intersection = len(set(section_tokens) & set(text_tokens))
            clone_prob = float(count_intersection)/len(section_tokens)
            #print(count_intersection, len(section_tokens), len(text_tokens), clone_prob)
            if(clone_prob > 0.5):
                #print(section, text)
                return True
    return False

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def search_url(db, url):
    cur = db.cursor()
    cur.execute("""select file, pos from urls_pos where url="{}";""".format(url))
    rows = cur.fetchall()
    if(len(rows) > 0):
        file_name, pos = rows[0]
        #print(pos)
        return file_name, pos
    return None, None

def find_url_content(url, path, db):
    #print(index)
    file_name, pos = search_url(db, url)
    if(file_name is not None):
        with open('{}{}'.format(path,file_name)) as file:
            for position, line in enumerate(file):
                if(position == pos):
                    try:
                        content = json.loads(line)
                        return content['sentences']
                    except:
                        print(line)
                        raise
    return None

def main(args):
    count = 0
    dirs = ['AA/', 'AB/']
    db = create_connection('data/brwac_ref_urls_sentences/pos.db')
    for dir_name in dirs:
        file_names = [f for f in listdir(args.brwac_urls_path1 + dir_name) if isfile(join(args.brwac_urls_path1 + dir_name, f))]
        for file_name in file_names:
            print(file_name)
            with open(args.brwac_urls_path1 + dir_name + file_name, 'r') as file1:
                with open(args.wiki_path + dir_name + file_name, 'r', encoding="utf-8") as wiki_file:
                    try:
                        with open(args.brwac_urls_path2 + dir_name + file_name, 'r') as file2:
                            for line in file1:
                                urls = {}
                                content1 = json.loads(line)
                                content2 = json.loads(file2.readline())
                                wiki_content = json.loads(wiki_file.readline())
                                for url in content1['urls']:
                                    urls[url] = None
                                for url in content2['urls']:
                                    urls[url] = None
                                if(len(urls) >= 1):
                                    count = count + 1
                                    print(count)
                                    to_write_urls = {}
                                    for url in urls:
                                        if(len(to_write_urls) >= 50):
                                            break
                                        url_content = find_url_content(url, args.urls_sentences_path, db)
                                        if(url_content is not None):
                                            to_write_urls[url] = url_content
                                    with open(args.out_file_path + 'wiki.json', 'ab+') as out_file:
                                        to_out = '{}\n'.format(json.dumps({'wiki_id' : content1['id'], 'wiki_title' : wiki_content['title'], 'wiki_sections' : wiki_content['sections'], 'wiki_text' : wiki_content['text']}, ensure_ascii=False)).encode('utf-8')
                                        #print(to_out)
                                        out_file.write(to_out)
                                    with open(args.out_file_path + 'brwac.json', 'ab+') as out_file:
                                        to_out = '{}\n'.format(json.dumps({'wiki_id' : content1['id'], 'wiki_title' : wiki_content['title'], 'urls' : to_write_urls}, ensure_ascii=False)).encode('utf-8')
                                        #print(to_out)
                                        out_file.write(to_out)
                    except:
                        for line in file1:
                            urls = {}
                            content1 = json.loads(line)
                            wiki_content = json.loads(wiki_file.readline())
                            for url in content1['urls']:
                                urls[url] = None
                            if(len(urls) >= 1):
                                count = count + 1
                                print(count)
                                to_write_urls = {}
                                for url in urls:
                                    if(len(to_write_urls) >= 50):
                                        break
                                    url_content = find_url_content(url, args.urls_sentences_path, db)
                                    if(url_content is not None):
                                        to_write_urls[url] = url_content
                                with open(args.out_file_path + 'wiki.json', 'ab+') as out_file:
                                    to_out = '{}\n'.format(json.dumps({'wiki_id' : content1['id'], 'wiki_title' : wiki_content['title'], 'wiki_sections' : wiki_content['sections'], 'wiki_text' : wiki_content['text']}, ensure_ascii=False)).encode('utf-8')
                                    #print(to_out)
                                    out_file.write(to_out)
                                with open(args.out_file_path + 'brwac.json', 'ab+') as out_file:
                                    to_out = '{}\n'.format(json.dumps({'wiki_id' : content1['id'], 'wiki_title' : wiki_content['title'], 'urls' : to_write_urls}, ensure_ascii=False)).encode('utf-8')
                                    #print(to_out)
                                    out_file.write(to_out)
    print(count)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Count the number of examples with more then one reference web page.')
	parser.add_argument('--wiki_path', default='data/wikipedia_articles_json/', type=str)
	parser.add_argument('--brwac_urls_path1', default='data/wikipedia_ref_urls_brwac/', type=str)
	parser.add_argument('--brwac_urls_path2', default='data/wikipedia_ref_urls_brwac_TITLES/', type=str)
	parser.add_argument('--urls_sentences_path', default='data/brwac_ref_urls_sentences/', type=str)
	parser.add_argument('--out_file_path', default='data/full_examples2/', type=str)
	args = parser.parse_args()
	main(args)
