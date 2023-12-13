import nltk
nltk.download('punkt')
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from nltk import sent_tokenize
import math, re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchmetrics.text.rouge import ROUGEScore
from transformers import Trainer, TrainingArguments, pipeline
import argparse
import evaluate
# from styleformer import Styleformer
import warnings
warnings.filterwarnings("ignore")
import copy
import multiprocessing
import pickle as pkl
import openai
from dotenv import load_dotenv
import os
from time import sleep

#Util functions
def get_overlap_scores(sentences, document):
    corpus = sentences + document
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus)
    similarities = (tfidf * tfidf.T).toarray()
    
    return similarities[:len(sentences), len(sentences):]


def get_summary_indices(article, summary, top_k=1, tolerance=0.1):
    scores = get_overlap_scores(summary, article)
    idx = scores.argmax(axis=1)
    false_idxs = np.where(scores.max(axis=1) == 0)
    idx = np.delete(idx, false_idxs)
    scores = np.delete(scores, false_idxs, axis=0)

    if top_k > 1 and len(article) > 1:
        search_idx = np.where((scores.max(axis=1) < 1-tolerance))
        biggest_idx = np.argpartition(scores[search_idx], -top_k)[:, -top_k:]
        unique_idx = np.concatenate((idx, biggest_idx.flatten()))
        unique_idx = np.unique(unique_idx)
    else:
        unique_idx = np.unique(idx)
    
    unique_idx.sort()

    return unique_idx



print("-"*20)

def generate_n_segments(a, n=10): #NEW
  k, m = divmod(len(a), n)
  return list((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))

def get_paraphrased(input_text, api_key):
    target_sentences = []
    
    # return changed sentences in list target sentences
    
    openai.api_key = api_key
    

    
    for idx,sentence in enumerate(input_text):
        # print(idx,title)
        
        # print(input_text)
        while True:
            try:
                response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "system", "content": 'You are a helpful assistant that is an expert in paraphrasing sentences.'},
                                        {"role": "user", "content": """Paraphrase the sentence I will provide. Please respond with just the paraphrased version of the sentence. Here is the sentence:
                                         
                                        
    {}

                                    
                                        """.format(sentence)}
                            ])

                
                res=response["choices"][0]["message"]["content"]
                # print(res)
                target_sentences.append(res)
                sleep(3)
                break
            except Exception as e:
                sleep(3)
                print(e)
    
    
    
    
    
    
    
#     if args.style=="deep1":
#         context = ["transfer Active to Passive: "+ text for text in input_text]

#     elif args.style=="deep2":
#         context = ["paraphrase: "+ text + " </s>" for text in input_text]

#     encoding = tokenizer(context,max_length=model.config.max_length, padding=True, return_tensors="pt")
#     input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
#     model.eval()
#     generations = model.generate(
#         input_ids=input_ids,attention_mask=attention_mask,
#         max_length=model.config.max_length,
#         early_stopping=True,
#         num_beams=5,
#         num_beam_groups = 1,
#         num_return_sequences=1,
#   )
#     for beam_output in generations:
#       sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#       target_sentences.append(sent)

    return target_sentences

def tokenize(example):
    example["tokenized_document"] = nltk.sent_tokenize(example[article_key])
    example["tokenized_summary"] = nltk.sent_tokenize(example[summary_key])
   # example['segment_idxs'] = generate_n_segments(example["article"]) #NEW
    return example

def tokenize_news(example):
    example["tokenized_document"] = nltk.sent_tokenize(example[article_key])
    example["tokenized_summary"] = nltk.sent_tokenize(example[summary_key][0][article_key])
    #example['segment_idxs'] = generate_n_segments(example["article"]) #NEW
    return example


def generate_paraphrased_articles(dataset, dataset_name, api_key,batch_size=1):
    
    if dataset_name!='news':
        dataset = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    else:
        dataset = dataset.map(tokenize_news, num_proc=multiprocessing.cpu_count())
        
    articles = dataset["tokenized_document"]
    highlights = dataset["tokenized_summary"]

    pp_articles = []

    assert len(articles) == len(highlights), "Error in dataset. Unequal lengths."
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_articles = articles[i:min(i+batch_size, len(articles))]
        batch_highlights = highlights[i:min(i+batch_size, len(articles))]
        batch_pp_articles = []
        batch_sentences = []
        batch_idx = []
        separator = 'X'
        collected=os.listdir(f"saved_data_new/{dataset_name}")
        
        if "{}.pkl".format(i) in collected:
            continue

        for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):

            idx = get_summary_indices(article, summ, top_k=2, tolerance=0.1)
            
            sentences = [article[x] for x in idx]
            # if (idx != 'A').all():
            #     sentences = [article[x] for x in idx]
            # else:
            #     sentences = ['No']

            batch_idx.extend(list(idx))
            batch_idx.append(separator)
            
            batch_sentences.extend(sentences) 
            # batch_sentences.append(separator)           
        
    
        paraphrase = get_paraphrased(batch_sentences,api_key)
        index = 0

        for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):
            paraphrased_article = article
            while index < len(batch_idx) and batch_idx[index] != separator and batch_idx[index] != 'A':
                try:
                    paraphrased_article[batch_idx[index]] = paraphrase[index]
                except:
                    print("In except.")
                    print(paraphrased_article)
                    print(index)
                    print(batch_idx[index])
                index += 1
            index += 1
            
            batch_pp_articles.append(' '.join(paraphrased_article))

        # pp_articles.extend(batch_pp_articles)
        with open(f'saved_data_new/{dataset_name}/{i}.pkl', 'wb') as f:
            pkl.dump(batch_pp_articles, f)
    
    return pp_articles


if __name__=='__main__':
    load_dotenv()
    api_key = os.getenv("API_KEY")

    dataset = load_dataset("cnn_dailymail", '3.0.0')
    article_key = 'article'
    summary_key = 'highlights'
    name='cnn_dailymail'
    dataset=dataset['test']
    
    # dataset=dataset.select(range(10))

    name='cnn'

    paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)
    # Save data
    
    
    dataset = load_dataset("xsum")
    article_key = 'document'
    summary_key = 'summary'
    dataset=dataset['test']


    name='xsum'
    
    # dataset=dataset.select(range(10))

    

    paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)
    
    dataset = load_dataset("argilla/news-summary")
    article_key = 'text'
    summary_key = 'prediction'
    dataset = DatasetDict({
        'train': dataset['test'],
            'test': dataset['train']})
    dataset=dataset['test']

    name='news'
    
    paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)

    # dataset=dataset.select(range(10))
    

    dataset = load_dataset('reddit_tifu', 'long')
    article_key = 'documents'
    summary_key = 'tldr'
    # 80% train, 20% test + validation
    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})

    name='reddit'

    # dataset=dataset.select(range(10))
    

    paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)

