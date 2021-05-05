import json
import threading, queue
import tensorflow as tf
import argparse
import time
import random
import re
from urllib.request import urlopen
import pickle
#from bs4 import BeautifulSoup
#import html2text
import htmlparser
from google.cloud import storage


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
'''
text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_tables = True
text_maker.ignore_images = True
text_maker.ignore_anchors = True
text_maker.ignore_emphasis = True
text_maker.body_width = 0
'''
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
	# Expect a minimum number of words.
	tokens = p.split()
	if len(tokens) < 6:
		#print(tokens, 'aqui')
		return True
	# Require some letters.
	if not re.search(_SOME_ALPHA_RE, p):
		#print(tokens, 'aqui1')
		return True
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
		return True
	#print(tokens, 'aqui3')
	return False

def detect_clone(text, wiki_articles):
	return False

def get_page_text(url):
	to_save = []
	try:
		url_fetch = urlopen(url, timeout=3)
		url_bytes = url_fetch.read()
		html_str = url_bytes.decode("utf8")
		#print(html_str)
		#start_bsoup = time.time()
		#textv1 = bsoup_parse(html)
		#end_bsoup = time.time()
		#text = html2text_parse(html_str)
		#print(text)
		text = htmlparser.get_text_from_html(html_str)
		if not detect_clone(text, None):
			for paragraph in text.split('\n'):
				if not filter_paragraph(paragraph):
					#print(paragraph)
					to_save.append(paragraph)
		#print(to_save)
		#end_html2text = time.time()
		#print((end_bsoup - start_bsoup), (end_html2text - end_bsoup))
		#with open("BSOUP.txt", 'a+') as file:
		#	file.write(textv1)
		#with open("HTML2TEXT.txt", 'a+') as file:
		#	file.write(textv2)
		#print(text)
	except:
		pass
	finally:
		return to_save

def worker():
	while True:
		item = q.get()
		#print("Working on {}".format(item))
		# time.sleep(random.randint(1, 4)) # simulando tempo
		extension = item.split('.')[-1]
		#print(extension)
		if(extension not in ['pdf', 'mp3', 'mp4']):
			paragraphs = get_page_text(item)
			if(len(paragraphs) is not 0):
				inputs_to_store.put([item, paragraphs])
				#print([item, paragraphs])
			#print(f'Finished {item}')
		q.task_done()

def persist_data(inputs, file_path, from_queue=False):
	itens_dict = {}
	if(from_queue):
		for item in drain(inputs):
			itens_dict[item[0]] = item[1]
	else:
		itens_dict[inputs[0]] = inputs[1:]
	with tf.io.gfile.GFile(file_path, "a+") as file:
		file.write(json.dumps(itens_dict) + '\n')
	#for item in itens_dict:
	#	print(itens_dict)

def get_content(file_name):
    #print(file_name)
    raw_dataset = tf.data.TFRecordDataset([file_name])
    #print(raw_dataset)
    wiki_articles = {}
    for raw_record in raw_dataset: # .take(2):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        title = example.features.feature['title'].bytes_list.value[0].decode('utf-8', errors='ignore')
        url = example.features.feature['url'].bytes_list.value[0].decode('utf-8', errors='ignore')
        texts = []
        section_titles = []
        for i in range(len(example.features.feature['section_texts'].bytes_list.value)):
            texts.append(example.features.feature['section_texts'].bytes_list.value[i].decode('utf-8', errors='ignore'))
            section_titles.append(example.features.feature['section_titles'].bytes_list.value[i].decode('utf-8', errors='ignore'))
        wiki_articles[title] = [title, url, section_titles, texts]
        #print(wiki_articles[title])
    return wiki_articles

def main(args):
	shard_urls_file = '{}wiki_urls.json-{:05d}-of-01000'.format(args.urls_path, args.shard_id)
	shard_content_file = '{}wiki_content.tfrecords-{:05d}-of-01000'.format(args.content_path, args.shard_id)
	shard_inputs_file = '{}inputs.txt-{:05d}-of-01000'.format(args.output_path, args.shard_id)
	shard_outputs_file = '{}outputs.txt-{:05d}-of-01000'.format(args.output_path, args.shard_id)
	#print(shard_urls_file, shard_content_file)
	total_start = time.time()
	wiki_articles = get_content(shard_content_file)
	with tf.io.gfile.GFile(shard_urls_file, "r") as urls_file:
		wiki_urls = json.loads(urls_file.readline())
		total_wikis = len(wiki_urls)
		i = 1
		for wiki_article in wiki_urls:
			start = time.time()
			new_entry = wiki_urls[wiki_article]
			refs = new_entry[u'refs']
			print("Processing {}".format(new_entry['title']))
			print("Sending {:d} URLs to workers".format(len(refs[:20])))
			# send thirty task requests to the worker
			for item in refs[:20]:
				q.put(item)
			# block until all tasks are done
			q.join()
			if(inputs_to_store.qsize() is not 0 and new_entry['title'] in wiki_articles):
				persist_data(inputs_to_store, shard_inputs_file, from_queue=True)
				persist_data(wiki_articles[new_entry['title']], shard_outputs_file, from_queue=False)
			#persist_data()
			end = time.time()
			print('{:d}/{:d} completed in {:.2f} seconds.'.format(i, total_wikis, (end-start)))
			i = i + 1
	total_end = time.time()
	print('Shard {:d} - FINAL TIME: {:.2f}'.format(args.shard_id, (total_end-total_start)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Wikisum dataset from given URLs and Wikipedia descriptions.')
	parser.add_argument('--shard_id', help='shard id (0 to 1000)', default=0, type=int) # 0 ate 1000
	parser.add_argument('--workers', help='number of threads to perform web searchs in parallel', default=1, type=int)
	parser.add_argument('--urls_path', default='wikisum-en/wiki_urls/', type=str)
	parser.add_argument('--content_path', default='wikisum-en/wiki_content/', type=str)
	parser.add_argument('--output_path', default='wikisum-en/txt/', type=str)
	args = parser.parse_args()
	# turn-on the worker thread
	for i in range(args.workers):
		threading.Thread(target=worker, daemon=True).start()
	main(args)
