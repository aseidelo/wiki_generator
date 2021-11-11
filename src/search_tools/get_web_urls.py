from codecs import open
from googlesearch import search
import json
import threading, queue

'''
Input:
	Extracted Wikimedia dump txt of articles titles and ids: 
        ptwiki-20210320-pages-articles-multistream-index.txt

	Output: csv file with (article_id, article_title, [url1, ..., urln])
'''

def check_restrictions(url):
    restrictions = ['.pdf', '.mp4', '.jpeg', '.jpg']# ['wikipedia.org', '.pdf', '.mp4', '.jpeg', '.jpg']
    for restriction in restrictions:
        if(restriction in url):
            return False
    return True

q_in = queue.Queue()
q_out = []

def worker():
    while True:
        article  = q_in.get()
        urls = search(article[1], lang="pt-br")#num_results = 15, lang="pt-br")
        good_urls = []
        query = ''
        for url in urls:
            if('/search?q=' in url):
                query = url
            elif (check_restrictions(url)):
                good_urls.append(url)
        q_out.append([article[0], article[1], good_urls, query])
        q_in.task_done()
        #print(f'Finished {item}')

def get_articles(file_path, out_path):
    '''
    already_done_articles = []
    try:
        with open(out_path, 'r') as file:
            for line in file:
                article = json.loads(line)
                already_done_articles.append(article['id'])
    except:
        pass
    '''
    last_id = 0
    try:
        with open(out_path, 'r') as file:
            for line in file:
                article = json.loads(line)
                if(int(article['id']) > last_id):
                    last_id = int(article['id'])
    except:
        pass
    docs = []
    print('Loading articles')
    with open(input_path + input_file, 'r') as file:
        for line in file:
            #print('{}/{}'.format(i, 105695))
            attrib = line.split(':')
            article_id = attrib[1]
            article_title = attrib[2].replace('\n', '')
            if(int(article_id) > last_id):
                if('(desambiguação)' not in article_title):
                    docs.append([article_id, article_title])
    return docs

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def search_urls(wiki_articles, workers = 1, n=15):
    global q_in, q_out
    articles_and_urls = []
    print('Searching urls:')
    if(workers==1):
        i = 1
        for article in wiki_articles:
            print('({}/{})'.format(i, len(wiki_articles)))
            article_id = article[0]
            article_title = article[1]
            urls = search(article_title, num_results = n, lang="pt-br")
            good_urls = []
            query = ''
            for url in urls:
                if('/search?q=' in url):
                    query = url
                elif (check_restrictions(url)):
                    good_urls.append(url)
            #print('title: {} id: {} \n {} urls: {}'.format(article_title, article_id, len(good_urls), good_urls))
            articles_and_urls.append([article_id, article_title, good_urls, query])
            i = i + 1
    elif(workers > 1):
        for article in wiki_articles:
            q_in.put(article)
        # block until all tasks are done
        q_in.join()
        for article_and_urls in q_out:
            articles_and_urls.append(article_and_urls)
        q_out = []
    return articles_and_urls

def store_articles_and_urls(articles_and_urls, output_path):
    with open(output_path, 'ab+') as file:
        for article in articles_and_urls:
            doc = {'id' : article[0], 'title' : article[1], 'n_urls' : len(article[2]), 'urls' : article[2], 'query' : article[3]}
            to_out = json.dumps(doc, ensure_ascii=False).encode('utf-8')+'\n'.encode('utf-8')
            file.write(to_out)

def search_and_store_urls(wiki_articles, output_path, batch_size, workers = 1):
    n_chunks = int(len(wiki_articles)/batch_size + 0.5)
    i = 1
    for batch in chunks(wiki_articles, batch_size):
        print('batch {}/{}'.format(i, n_chunks))
        articles_and_urls = search_urls(batch, workers)
        store_articles_and_urls(articles_and_urls, output_path)
        i = i + 1

if __name__ == '__main__':
    input_path = 'dumps/index/'
    input_file = 'ptwiki-20210320-pages-articles-multistream-index3.txt-p513713p1629224'
    output_path = 'urls/'
    output_file = 'p513713p1629224.json'
    batch_size = 10
    n_workers = 10
    '''
    ptwiki-20210320-pages-articles-multistream-index1.txt-p1p105695
    ptwiki-20210320-pages-articles-multistream-index2.txt-p105696p513712
    ptwiki-20210320-pages-articles-multistream-index3.txt-p513713p1629224
    ptwiki-20210320-pages-articles-multistream-index4.txt-p1629225p2880804
    ptwiki-20210320-pages-articles-multistream-index5.txt-p2880805p4380804
    ptwiki-20210320-pages-articles-multistream-index5.txt-p4380805p5024908
    ptwiki-20210320-pages-articles-multistream-index6.txt-p5024909p6524729
    '''
    articles = get_articles(input_path+input_file, output_path + output_file)
    if(n_workers > 1):
        for i in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()
    search_and_store_urls(articles, output_path + output_file, batch_size, n_workers)
    #store_articles_and_urls(articles_and_urls, output_path + output_file)
