from codecs import open
import xml.etree.cElementTree as ET
import json
import re
from os import listdir
from os.path import isfile, join

'''
Input:
	set of files with Wikipedia txt articles generate by wikiextractor 
	(<https://github.com/attardi/wikiextractor>) from wikimedia dump with:

	wikiextractor --no-templates -o .  ../ptwiki-20210320-pages-articles-multistream.xml.bz2 -b 10M

	Output: set of files with same naming from input with json wikipedia articles separed by sections
'''

input_path = 'wikiextractor/'
input_dirs = ['AA/', 'AB/']
output_path = 'processed_wikiextractor/'
dirs_i = 1
i = 1
for input_dir in input_dirs:
	file_names = [f for f in listdir(input_path+input_dir) if isfile(join(input_path+input_dir, f))]
	for file_name in file_names:
		print('({}/{}) - {}/{}'.format(dirs_i, len(input_dirs), i, len(file_names)))
		xml = ''
		with open(input_path + input_dir + file_name, 'r', encoding="utf-8") as f:
			xml = f.read()
		tree = ET.fromstring("<root>" + xml + "</root>")
		#print(tree)
		docs = []
		for doc in tree.findall('doc'):
			content = doc.text.rstrip()
			splited_content = content.split('\n\n')
			title = doc.attrib['title']
			doc_id = doc.attrib['id']
			try:
				paragraphs = splited_content[1]
				doc_dict = {'title' : title, 'id' : doc_id, 'text' : [''], 'sections' : ['']}
				splited_paragraphs = paragraphs.split('\n')
				for sentence in splited_paragraphs:
					sent = sentence.rstrip()
					n_words = len(sent.split(' '))
					if(n_words >= 4):
						if(doc_dict['text'] == ['']):
							doc_dict['text'][-1] = doc_dict['text'][-1] + sent
						else:
							doc_dict['text'][-1] = doc_dict['text'][-1] + '\n' + sent
					elif(sent != ''):
						doc_dict['sections'].append(sent)
						doc_dict['text'].append('')
				docs.append(doc_dict)
			except:
				pass
		with open(output_path + input_dir + 'processed_{}.json'.format(file_name), 'wb+') as file:# , encoding='utf-8', errors='strict') as file:
			for doc in docs:
				file.write(json.dumps(doc, ensure_ascii=False).encode('utf-8')+'\n'.encode('utf-8'))
		i = i + 1
	i = 0
	dirs_i = dirs_i + 1
		# with open(input_path + file_name, 'r') as file:
		#	for line in file:
		#		print(line)
