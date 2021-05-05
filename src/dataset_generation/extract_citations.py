#!/usr/bin/env python2
"""
# Extract citations from Wikipedia dumps
This script makes a best-effort at resolving inline references and citations.
Wikipedia citations are complicated, so this is a quick and dirty hack.
It's intented for research in resolving citations and natural language processing.

Similar tools:

 - https://github.com/mediawiki-utilities/python-mwcites


## Example usage
    curl "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" | bzcat | python extract_citations.py

## Output format
One JSON object per line, structured like this:

    {
        "page": "Page Title",
        "paragraphs": [
            {
                "text":"The plaintext paragraph (with mediawiki style tags).",
                "refs":[
                    { "cite": 0, "offset":0},
                    { "cite": 0, "offset":0, ... }
                ]
            }
        ],
        "reflist": [
            {
                "text citation",
                {"@type":"template_name", ...}
            },
        ]
    }


## How Wikipedia cites sources
(see also: https://en.wikipedia.org/wiki/Wikipedia:Citing_sources)
References can either be structured using templates like `{{cite}}``, or be plain text.

Inline with text, references can have the following forms:

 - `<ref>` tags
 - `{{r}}` or `{{rp}}` templates (TODO)

The `{{cite}}` templates are typically in a References section or inside the `<ref>` tag.
The `<ref>` tags generate references lists that are displayed using `{{reflist}}`.
This list can contain `{{cite}}` templates itself (using `refs=`). (TODO!)

# TODO

 - resursive template parsing
 - short ref templates

"""
from __future__ import print_function
import sys, itertools, re
from collections import OrderedDict
from xml.etree.cElementTree import iterparse
import mwparserfromhell
import HTMLParser

FULL_CITE_TEMPLATES = ['cite', 'citation', 'vcite2', 'vcite', 'vancite', 'wikicite', 'wayback']
SHORT_CITE_TEMPLATES = ['sfn','sfnp','sfnm','harvnb']

# Short citation parsing is hard.
# https://en.wikipedia.org/wiki/Wikipedia:Citation_templates_and_reference_anchors
def last_part(x):
    x = x.split()
    return x[-1] if x else ''

CITEREF_PARAM_OPTIONS = [
    (r'last\d?', None),
    (r'surname\d?', None),
    (r'author\d?', None),
    (r'authors', None),
]
CITEREF_PARAM_OPTIONS_EDITOR = [
    (r'editor\d?-last', None),
    (r'editor\d?-surname', None),
    (r'editor\d?', last_part),
    (r'editors', None),
]
CITEREF_PARAM_OPTIONS_YEAR = [
    (r'date', last_part),
    (r'year', None),
    (r'publication-date', last_part)
]

def find_params(p, preprocess, cite, anchor):
    for param in cite:
        if re.match('%s$' % p, param):
            if preprocess:
                val = preprocess(cite[param])
            else:
                val = cite[param]
            if val:
                anchor.append(unicode(val))

def make_citeref_anchor(cite):
    anchor = []
    for p, preprocess in CITEREF_PARAM_OPTIONS:
        find_params(p, preprocess, cite, anchor)
    if not anchor:
        for p, preprocess in CITEREF_PARAM_OPTIONS_EDITOR:
            find_params(p, preprocess, cite, anchor)
    for p, preprocess in CITEREF_PARAM_OPTIONS_YEAR:
        find_params(p, preprocess, cite, anchor)
    return tuple(anchor)



def is_full_cite(i):
    for template in FULL_CITE_TEMPLATES:
        if i.name.lower().startswith(template):
            return True
    return False 

def is_short_cite(i):
    for template in SHORT_CITE_TEMPLATES:
        if i.name.lower().startswith(template):
            return True
    return False

def template_to_dict(template):
    cite_type = template.name.strip().replace(' ', '_')
    cite = OrderedDict([('@type', cite_type)])
    for param in template.params:
        val = param.value.strip_code().strip()
        if val:
            cite[param.name.strip()] = val
    return cite


class References():
    
    def __init__(self, wikipage):
        self.wikipage = wikipage
        self.citelist = []
        # keys are names from <ref name=Foot01/>, values are list of cites
        self.cite_notes = {}
        # keys are tuples? from {{sfn|Miller|2005}} etc), values are cites
        self.citeref = {}

    def add_cite(self, template):
        cite = template_to_dict(template)
        cite_index = len(self.citelist)

        # See if it's in citeref
        key = make_citeref_anchor(cite)
        if key:
            if key in self.citeref:
                cite_index = self.citeref[key]
                self.citelist[cite_index].update(cite)
                return cite_index, {}
            else:
                self.citeref[key] = cite_index

        # Append to cite list
        self.citelist.append(cite)
        return cite_index, {}

    def add_text_cite(self, text):
        cite_index = len(self.citelist)
        self.citelist.append(text)
        return cite_index, {}
    
    def add_link_cite(self, title, url):
        cite_index = len(self.citelist)
        self.citelist.append(OrderedDict([
            ('@type', 'link'), 
            ('title', unicode(title)), 
            ('url', unicode(url))
        ]))
        return cite_index, {}

    def short_cite(self, cite_node):
        """Resolve short cite references"""
        key = tuple(unicode(p.value) for p in cite_node.params if not p.showkey)
        if key in self.citeref:
            cite_index = self.citeref[key]
        else:
            cite_index = len(self.citelist)
            self.citelist.append(OrderedDict())
            self.citeref[key] = cite_index
        params = {unicode(p.name): unicode(p.value) 
            for p in cite_node.params if p.showkey}
        return cite_index, params

    def cites_from_ref_tag(self, ref_node):
        """Resolve references, return reflist index"""
        if ref_node.has('name'):
            name = unicode(ref_node.get('name').value)
            name = name.strip().encode('utf8')
            if name not in self.cite_notes:
                # Add emtpy reference if not present
                self.cite_notes[name] = []
            cites = self.cite_notes[name]
            if ref_node.self_closing:
                return cites
        else:
            cites = []

        # Let's assume that nobody's crazy enough to use templates in footnotes
        if ref_node.contents:
            # Find cite templates in ref
            for i in ref_node.contents.ifilter_templates(recursive=False):
                if is_full_cite(i): 
                    cites.append( self.add_cite(i) )
                elif is_short_cite(i):
                    cites.append( self.short_cite(i) )
            # Find text in ref
            text = ref_node.contents.strip_code().strip()
            text = ' '.join(s for s in text.split() if s != ';')
            if text:
                cites.append( self.add_text_cite(text) )
            # Find links in ref
            for i in ref_node.contents.ifilter_external_links(recursive=False):
                cites.append( self.add_link_cite(i.title, i.url) )
        if not cites:
            print(self.wikipage, ': no cites in:\n\t', unicode(ref_node), file=sys.stderr)

        return cites

    
    def parse_code(self, wikicode):
        """Run through some wikicode, yield text chunk and ref_index"""
        text_chunk = ''
        for i in wikicode.ifilter(recursive=False):
            if type(i) == mwparserfromhell.nodes.Tag and i.tag == 'ref':
                # Add reference
                yield text_chunk, self.cites_from_ref_tag(i)
                text_chunk = ''
            elif type(i) == mwparserfromhell.nodes.Template:
                # Find citations in templates
                if is_full_cite(i):
                    yield text_chunk, [ self.add_cite(i) ]
                elif is_short_cite(i):
                    yield text_chunk, [ self.short_cite(i) ]
                else:
                    # TODO: recursive template parsing
                    # TODO: {{r}} template parsing
                    pass
            elif type(i) == mwparserfromhell.nodes.Wikilink:
                # Add the text of normal (non-image etc) wikilinks
                if not re.match('^[A-Z][a-z]+:', unicode(i.title)):
                    if i.text:
                        text_chunk += unicode(i.text)
                    else:
                        text_chunk += unicode(i.title)
            elif type(i) == mwparserfromhell.nodes.Text:
                i = unicode(i)
                # Split into paragraphs
                if '\n\n' in i:
                    end = i.index('\n\n')
                    text_chunk += i[:end]
                    yield text_chunk, None
                    text_chunk = i[end:]
                else:
                    text_chunk += i

    def parse_paragraphs(self, wikicode):
        """Get references in text"""
        text, refs = '', []
        for text_chunk, cites in self.parse_code(wikicode):
            # Add references
            if not text:
                text_chunk = text_chunk.lstrip()
            text += re.sub('\s+', ' ', text_chunk)
            if cites == None:
                # Split into paragraphs
                yield text.rstrip(), refs
                text, refs = '', []
            else:
                if cites:
                    for index, params in cites:
                        cite = OrderedDict([
                            ('offset', len(text)),
                            ('cite', index)
                            ])
                        cite.update(params)
                        refs.append(cite)
        yield text, refs
                           
        
        
if __name__ == '__main__':
    import json
    source = sys.stdin

    # Mediawiki XML dump parsing
    elems = (elem for _, elem in iterparse(source, events=("end",)))
    elem = next(elems)
    namespace = re.match("^{(.*?)}", elem.tag).group(1)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping

    # Mediawiki Markup parsing
    for elem in elems:
        if elem.tag.endswith('page'):
            title = elem.find(title_path).text.replace(' ','_')
            text = elem.find(text_path).text
            refs = References(title)
            paragraphs = []
            
            wikicode = mwparserfromhell.parse(text, skip_style_tags=True)
            for paragraph, cites in refs.parse_paragraphs(wikicode):
                if paragraph and cites:
                    paragraphs.append({'text': paragraph, 'refs':cites})
            
            if refs.citelist:
                print(json.dumps(OrderedDict([
                    ('page', title),
                    ('paragraphs', paragraphs),
                    ('references', refs.citelist),
                ])))
                for k,v in refs.citeref.items():
                    v = refs.citelist[v]
                    if not v:
                        print(title,': missing:\n\t', k, file=sys.stderr)