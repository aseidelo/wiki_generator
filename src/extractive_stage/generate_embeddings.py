from transformers import AutoModel 
from transformers import AutoTokenizer
import torch
import json
import numpy as np

# for each example:
# - load embeddings
# - cluster embeddings
# fine tune and validation of cluster algo. params.:
# compare summary sentences embeddings with input sentence
# choose cluster params. that minimizes the minimum distance cluster-summary sentence x n clusters
# return:
# - Nmax cluster embeddings and Nmax sentences closest to cluster centroid

def split_sentences(doc):
    to_out_docs = []
    for sentence in doc.split('.'):
        if (len(sentence.split(' ')) >= 3): # at least 5 tokens in 1 sentence
            to_out_docs.append(sentence.replace('\n', ''))
    return to_out_docs

def load_example(input_file_path, output_file_path):
    # load dataset
    with open(output_file_path, "r") as output_f:
        with open(input_file_path, "r") as input_f:
            for input_line in input_f:
                output_line = output_f.readline()
                input_titlesentences = input_line.split('</s>')
                title = input_titlesentences[0]
                input_sentences_str = input_titlesentences[1]
                input_sentences = input_sentences_str.split('<\s>')
                output_sentences = split_sentences(output_line)
                yield {'title' : title, 'input_sentences' : input_sentences, 'output_sentences' : output_sentences}


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def set_batch(lst, batch_size=512):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def forward(batch, tokenizer, model, device):
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

# generate embedding representations
def generate_embeddings(example, tokenizer, model, device, embeddings_inputs_file, embeddings_outputs_file):
    # Tokenize sentences
    # - input sentences
    # - title
    # - summary sentences
    # batches
    input_batches = set_batch(example['input_sentences']) 
    input_embeddings = None
    for batch in input_batches:
        new_embeddings = forward(batch, tokenizer, model, device).to('cpu')
        if input_embeddings == None:
            input_embeddings = new_embeddings
        else:
            input_embeddings = torch.cat((input_embeddings, new_embeddings), 0)
    torch.cuda.empty_cache()
    wrap_output = [example['title']] + example['output_sentences']
    output_batches = set_batch(wrap_output) # title is considered a sentence from the output
    output_embeddings = None
    for batch in output_batches:
        new_embeddings = forward(batch, tokenizer, model, device).to('cpu')
        if output_embeddings == None:
            output_embeddings = new_embeddings
        else:
            output_embeddings = torch.cat((output_embeddings, new_embeddings), 0)
    print(len(wrap_output), output_embeddings.shape)
    torch.cuda.empty_cache()
    # save in file
    np.save(embeddings_inputs_file, input_embeddings)
    np.save(embeddings_outputs_file, output_embeddings)

def generate_dataset_embeddings(input_file_path, target_file_path, embeddings_inputs_file_path, embeddings_outputs_file_path, checkpoint):
    device = torch.device("cuda:0")
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=False, max_len=100)
    model.to(device)
    with open(embeddings_inputs_file_path, 'wb') as embeddings_inputs_file:
        with open(embeddings_outputs_file_path, 'wb') as embeddings_outputs_file:
            for example in load_example(input_file_path, target_file_path):
                print(example['title'])
                generate_embeddings(example, tokenizer, model, device, embeddings_inputs_file, embeddings_outputs_file)
                # break

if __name__ == '__main__':
    # generating embeddings
    input_file_path = "../../data/wikisum_ptbr/train_test_split/input_train.csv"
    target_file_path = "../../data/wikisum_ptbr/train_test_split/output_train.csv"
    checkpoint = 'neuralmind/bert-base-portuguese-cased'
    embeddings_inputs_file_path = "../../data/extractive_stage/cluster_embeddings/inputs_bert-base-portuguese-cased_train.npy"
    embeddings_outputs_file_path = "../../data/extractive_stage/cluster_embeddings/outputs_bert-base-portuguese-cased_train.npy"
    generate_dataset_embeddings(input_file_path, target_file_path, embeddings_inputs_file_path, embeddings_outputs_file_path, checkpoint)

