import json
import lxml.etree as ET
from codecs import open
from os import listdir
from os.path import isfile, join
from pathlib import Path

def read_brwac_docs(buffer):
    #print(buffer)
    try:
        buffer_out = buffer[buffer.rindex('</doc>') + 6:]
        xml = '<root> ' + buffer[:buffer.rindex('</doc>') + 6].replace('<g/>', '').replace('\n', ' ').replace('<p>', '').replace('</p>', '\n') + ' </root>'
        #print(xml)
        parser = ET.XMLParser(encoding="utf-8", recover='True')
        tree = ET.fromstring(xml.encode('utf-8'), parser=parser)
        docs = tree.findall('doc')
        return docs, buffer_out
    except:
        return None, buffer


    to_out_docs = {'wiki_id' : wiki_id, 'wiki_title' : text, 'brwac_uris' : [], 'brwac_titles' : [], 'brwac_texts' : []}
    with open(brwac_file_path, 'r', encoding="utf-8") as file:
        buffer = file.read(100000)