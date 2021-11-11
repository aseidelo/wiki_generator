# -*- coding: utf-8 -*-
from codecs import open
import json
import threading, queue
import argparse
import time
import random
import re
from urllib.request import urlopen
import pickle
from bs4 import BeautifulSoup
from nltk import word_tokenize 
from nltk.util import ngrams
from collections import Counter
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
#import tensorflow as tf
#import numpy as np
import html2text
#import htmlparser


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

q = queue.Queue()
inputs_to_store = queue.Queue()

text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_tables = True
text_maker.ignore_images = True
text_maker.ignore_anchors = True
text_maker.ignore_emphasis = True
text_maker.body_width = 0

def drain(q):
	while True:
		try:
			yield q.get_nowait()
		except queue.Empty:  # on python 2 use Queue.Empty
			break

def bsoup_parse(html):
	soup = BeautifulSoup(html, features="html.parser")
	# kill all script and style elements
	for script in soup(["script", "style"]):
	    script.extract()    # rip it out
	# get text
	text = soup.get_text()
	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)
	return text

def html2text_parse(html):
	text = text_maker.handle(html)
	return text

 # Peguei do Wikisum: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/wikisum/utils.py
_SOME_ALPHA_RE = re.compile(r'[A-Za-z]+')
_ONLY_ALPHA_RE = re.compile(r'^[A-Za-z]*$')
def filter_paragraph(p):
	"""Simple filter to remove obviously bad paragraphs (bad text extraction).
	Note this needs to run very quickly as it is applied to every paragraph
	in the corpus, so nothing fancy! This whole method should be linear
	expected time in len(p).
	Args:
	p: string, paragraph
	Returns:
	True if we should remove the paragraph.
	"""
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy .
	p = re.sub(r"([?.!,¿])", r" \1 ", p)
	p = re.sub(r'[" "]+', " ", p)
	# substituir tudo por espaço exceto (a-z, A-Z, ".", "?", "!", ",", letras com acentos da lingua pt)
	p = re.sub(r"[^a-zA-ZçÇéêíáâãõôóúûÉÊÍÁÂÃÕÔÓÚÛ?.!,()0-9]+", " ", p).lower()
    # e depois colocar em caixa baixa
	p = p.strip()
	# Expect a minimum number of words.
	tokens = p.split()
	if len(tokens) < 6:
		#print(tokens, 'aqui')
		return True, p
	# Require some letters.
	if not re.search(_SOME_ALPHA_RE, p):
		#print(tokens, 'aqui1')
		return True, p
	# Keep this one at the end, probably the most complicated logic.
	# We try to detect sentences, which should have a minimum of 3 tokens
	# with only alphabetic characters.
	last = 0
	found_sentence = False
	num_alpha = 0
	for i, x in enumerate(tokens):
		if x == '.':
			if i - last > 3 and num_alpha >= 3:
				found_sentence = True
				break
			last = i
			num_alpha = 0
		if re.match(_ONLY_ALPHA_RE, x):
			#print('OIOIOIO')
			num_alpha += 1
	if not found_sentence:
		#print(tokens, 'aqui2')
		return True, p
	#print(tokens, 'aqui3')
	return False, p

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

def get_page_text(url, article_sections=None):
	to_save = []
	try:
		url_fetch = urlopen(url, timeout=3)
		url_bytes = url_fetch.read()
		html_str = url_bytes#.decode("utf-8")
		text = bsoup_parse(html_str)
		if article_sections != None:
			if detect_clone(text, article_sections):
				return to_save
		paragraphs = text.split('\n')
		for paragraph in paragraphs:
			not_good, processed_para = filter_paragraph(paragraph)
			if not not_good:
				lang = detect(processed_para)
				if(lang == 'pt'):
					to_save.append(processed_para)
	except Exception as e:
		pass
	finally:
		return to_save

def worker():
	while True:
		item = q.get()
		url = item[0]
		article_sections = item[1]
		extension = url.split('.')[-1]
		article_id = item[2]
		article_title = item[3]
		article_sections_titles = item[4]
		try:
			#print("Working on {}".format(item))
			# time.sleep(random.randint(1, 4)) # simulando tempo
			#print(extension)
			if(extension not in ['pdf', 'mp3', 'mp4', 'zip']):
				paragraphs = get_page_text(url, article_sections)
				#print(paragraphs)
				if(len(paragraphs) != 0):
					inputs_to_store.put([article_id, article_title, article_sections_titles, article_sections, paragraphs])
					#print([item, paragraphs])
				#print(f'Finished {item}')
		except Exception as e:
			#print(e)
			pass
		finally:
			q.task_done()

def get_articles_urls(urls_file, out_file):
	already_done_articles = 0
	try:
		with open(out_file, 'r') as file:
			for line in file:
				article = json.loads(line)
				if(int(article['id']) > already_done_articles):
					already_done_articles = int(article['id'])
				#already_done_articles.append(article['id'])
	except:
		pass
	docs = []
	print('Loading articles')
	with open(urls_file, 'r') as file:
		for line in file:
			#print('{}/{}'.format(i, 105695))
			attrib = json.loads(line)
			article_id = attrib['id']
			article_title = attrib['title']
			if(int(article_id) > already_done_articles):
				article_urls = attrib['urls']
				article_n_urls = attrib['n_urls']
				docs.append([article_id, article_title, article_n_urls, article_urls])
	return docs

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_article_text(input_path, article_id):
	article_text = []
	article_sections_titles = []
	for file_dir in ['AB/', 'AA/']:
		for i in range(99, -1, -1):
			file_name = '{}{}processed_wiki_{:02d}.json'.format(input_path, file_dir, i)
			try:
				with open(file_name, 'r') as file:
					first_article = json.loads(file.readline())
					if(int(article_id) < int(first_article['id'])):
						continue
					elif(int(article_id) == int(first_article['id'])):
						article_text = first_article['text']
						article_sections_titles = first_article['sections']
						break
					else:
						for line in file:
							new_article = json.loads(line)
							if(int(article_id) == int(new_article['id'])):
								article_text = new_article['text']
								article_sections_titles = new_article['sections']
								break
			except:
				pass
	return article_text, article_sections_titles

def persist_data(item, file):
	to_out = json.dumps(item, ensure_ascii=False).encode('utf-8')+'\n'.encode('utf-8')
	file.write(to_out)
	#for item in itens_dict:
	#	print(itens_dict)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_str_list(item_list):
	str_list = ""
	i = 0
	for item in item_list:
		if(i == 0):
			str_list = item
		else:
			str_list = str_list + "[sep]" + item
		i = i + 1
	return str_list.encode('utf-8')

def serialize_example(article_id, title, section_titles, section_texts, web_paragraphs):
	"""
	Creates a tf.train.Example message ready to be written to a file.
	"""
	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
	# data type.
	feature = {
		'id': _bytes_feature(article_id.encode('utf-8')),
		'title': _bytes_feature(title.encode('utf-8')),
		'section_titles': _bytes_feature(serialize_str_list(section_titles)),
		'section_texts': _bytes_feature(serialize_str_list(section_texts)),
		'web_paragraphs' : _bytes_feature(serialize_str_list(web_paragraphs))
	}
	# Create a Features message using tf.train.Example.
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def tf_write(item, file_path):
	with tf.io.TFRecordWriter(file_path) as writer:
		example = serialize_example(item[0], item[1], item[2], item[3], item[4])
		writer.write(example)

def check_url(url_str):
	forbiden_strs = ['google', 'wikipedia', 'wikimedia', 'youtube', 'PDF', 'pdf', 'ftp', 'FTP', 'xls']
	for forbiden in forbiden_strs:
		if forbiden in url_str:
			return False
	return True

def main(args):
	urls_file = '{}{}'.format(args.urls_path, args.urls_file)
	output_file1 = '{}{}-WEB'.format(args.output_path, args.urls_file)
	#output_file1 = '{}{}.tfrecord'.format(args.output_path, args.urls_file)
	output_file2 = '{}{}-WIKI'.format(args.output_path, args.urls_file)
	#print(shard_urls_file, shard_content_file)
	total_start = time.time()
	wiki_articles = get_articles_urls(urls_file, output_file1)
	n_chunks = int(len(wiki_articles)/args.batch_size + 0.5)
	i = 1
	for batch in chunks(wiki_articles, args.batch_size):
		start = time.time()
		for article in batch:
			article_id = article[0]
			article_title = article[1]
			n_urls = article[2]
			wiki_urls = article[3]
			#print(article_id)
			article_sections, article_sections_titles = find_article_text(args.wiki_articles_path, article_id)
			#print(len(article_sections))
			if(len(article_sections) > 0):
				# send thirty task requests to the worker
				actual_n_urls = 0
				for url in wiki_urls:
					if(check_url(url)):
						actual_n_urls = actual_n_urls + 1
						q.put([url, article_sections, article_id, article_title, article_sections_titles])
				print("Processing {}".format(article_title))
				print("Sending {:d} URLs to workers".format(actual_n_urls))
		# block until all tasks are done
		#print(q.qsize())
		q.join()
		#print('aaa')
		#print(inputs_to_store.qsize())
		#web_paragraphs = []
		if(inputs_to_store.qsize() != 0):
			to_outs1 = {}
			to_outs2 = {}
			for url_data in drain(inputs_to_store):
				article_id = url_data[0]
				article_title = url_data[1]
				article_sections_titles = url_data[2]
				article_sections = url_data[3]
				url_paragraphs = url_data[4]
				if(article_id not in to_outs1):
					to_outs1[article_id] = {'id' : article_id, 'title' : article_title, 'web_paragraphs' : []}
					to_outs2[article_id] = {'id' : article_id, 'title' : article_title, 'sections_titles': article_sections_titles, 'sections_texts' : article_sections}
				to_outs1[article_id]['web_paragraphs'] = to_outs1[article_id]['web_paragraphs'] + url_paragraphs
			with open(output_file1, "ab+") as file1:			
				with open(output_file2, "ab+") as file2:
					for article_id in to_outs1:
						web_data = to_outs1[article_id]
						wiki_data = to_outs2[article_id]
						if(len(web_data['web_paragraphs']) >= 10):
							#web_paragraphs = web_paragraphs + paragraphs
							persist_data(web_data, file1)
							persist_data(wiki_data, file2)
			#tf_write([article_id, article_title, article_sections_titles, article_sections, web_paragraphs], output_file1)
		end = time.time()
		print('{:d}/{:d} completed in {:.2f} seconds.'.format(i, n_chunks, (end-start)))
		i = i + 1
	total_end = time.time()
	print('File {} - FINAL TIME: {:.2f}'.format(args.urls_file, (total_end-total_start)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Wikisum dataset from given URLs and Wikipedia descriptions.')
	parser.add_argument('--workers', help='number of threads to perform web searchs in parallel', default=1, type=int)
	parser.add_argument('--batch_size', help='batch of articles between storages', default=10, type=int)
	parser.add_argument('--urls_path', default='wiki_urls_refs/', type=str)
	parser.add_argument('--wiki_articles_path', default='processed_wikiextractor/', type=str)
	parser.add_argument('--urls_file', default='p1p105695.json', type=str)
	parser.add_argument('--output_path', default='processed_examples/pt_en_v2/', type=str)
	args = parser.parse_args()
	# turn-on the worker thread
	for i in range(args.workers):
		threading.Thread(target=worker, daemon=True).start()
	main(args)
