import pandas as pd
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5EncoderModel
from datasets import load_dataset, load_metric, DatasetDict, Dataset, ClassLabel

def load_ptt5(path, model_name, checkpoint):
    global tokenizer
    global model
    # load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(path + model_name + '/' + checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(path + model_name + '/' + checkpoint, gradient_checkpointing=True, use_cache=False).to("cuda").half()
    model.config.min_length = 50

def load_ptt5_encoder(path, model_name, checkpoint):
    global tokenizer
    global model
    # load tokenizer and model encoder
    tokenizer = T5TokenizerFast.from_pretrained(path + model_name + '/' + checkpoint)
    model = T5EncoderModel.from_pretrained(path + model_name + '/' + checkpoint, gradient_checkpointing=True, use_cache=False).to("cuda").half()
    model.config.min_length = 50

def generate_answer(batch, max_length=768):
    inputs_dict = tokenizer(batch["document"], padding="max_length", max_length=max_length, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    #global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    #global_attention_mask[:, 0] = 1
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask)
    #print(predicted_abstract_ids)
    batch["predicted_summary"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    # to release gpu memory
    input_ids = None
    attention_mask = None
    torch.cuda.empty_cache()
    return batch

def run_batch(documents, titles, batch_size=6):
    test_df = pd.DataFrame({'document': documents, 'title' : titles})
    test_dataset = Dataset.from_pandas(df=test_df)
    result = test_dataset.map(generate_answer, batched=True, batch_size=batch_size)
    return result
