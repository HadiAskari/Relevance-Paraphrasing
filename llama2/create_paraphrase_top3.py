# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

from tqdm.auto import tqdm
from time import sleep
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import os

# os.environ['CUDA_VISIBLE_DEVICES']="1,2"
import pandas as pd
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
import os
from time import sleep


#Util functions
def get_overlap_scores(sentences, document):
    corpus = sentences + document
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus)
    similarities = (tfidf * tfidf.T).toarray()
    
    return similarities[:len(sentences), len(sentences):]


def get_summary_indices(article, summary, top_k=3, tolerance=0.1):
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


def generate_n_segments(a, n=10): #NEW
  k, m = divmod(len(a), n)
  return list((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))

def prompt(sentence):

    prompt="""You are a helpful assistant that is an expert in paraphrasing sentences.
    Paraphrase the sentence I will provide. Please respond with just the paraphrased version of the sentence. Here is the sentence:
    
{}

""".format(sentence)

    return prompt



def build(ckpt_dir,tokenizer_path,max_seq_len,max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator


def get_paraphrased(generator, batch_sentences, max_gen_len,temperature,top_p):
    target_sentences = []
    
    for sent in batch_sentences:
        dialogs: List[Dialog] = []

        user_input = prompt(sent)

        dialogs.append([{"role": "user", "content": "{}".format(user_input)}])
            
    

        
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )


        for result in results:

            
            final=result['generation']['content']

        try:
            final=final.split(':')[1].split('\n\n')[1]
            # print(final)
            target_sentences.append(final)
        except Exception as e:
            print(e)
            print(final)
            return "-1"
        
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


def generate_paraphrased_articles(generator,dataset, dataset_name, max_gen_len,temperature,top_p,batch_size=1):
    
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
        collected=os.listdir(f"../rebuttal_experiments/Top3-Paraphrase/{dataset_name}")
        
        if "{}.pkl".format(i) in collected:
            continue

        for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):
            
            flag=False
          
            try:
                idx = get_summary_indices(article, summ, top_k=3, tolerance=0.1)
                print(idx)
            except:
                flag=True
                break
                
                
                
            sentences = [article[x] for x in idx]
            # if (idx != 'A').all():
            #     sentences = [article[x] for x in idx]
            # else:
            #     sentences = ['No']

            batch_idx.extend(list(idx))
            batch_idx.append(separator)
            
            batch_sentences.extend(sentences) 
            # batch_sentences.append(separator)           
        
        if flag==True:
            print('bad idx')
            with open(f'../rebuttal_experiments/Top3-Paraphrase/{dataset_name}/{i}.pkl', 'wb') as f:
                pkl.dump([], f)
            continue
            
    
        paraphrase = get_paraphrased(generator,batch_sentences,max_gen_len,temperature,top_p)
        
        if paraphrase == "-1":
            print('bad')
            with open(f'../rebuttal_experiments/Top3-Paraphrase/{dataset_name}/{i}.pkl', 'wb') as f:
                pkl.dump([], f)
            continue
        
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
        with open(f'../rebuttal_experiments/Top3-Paraphrase/{dataset_name}/{i}.pkl', 'wb') as f:
            pkl.dump(batch_pp_articles, f)
    
    return pp_articles





    
 
    

if __name__ == "__main__":
    # fire.Fire(main)
    ckpt_dir= 'llama-2-13b-chat/'
    tokenizer_path= 'tokenizer.model'
    temperature: float = 0.7
    top_p: float = 0.9
    max_seq_len: int = 2048
    max_batch_size: int = 1
    max_gen_len: Optional[int] = None


    generator = build(ckpt_dir,tokenizer_path,max_seq_len,max_batch_size)
    
    # dataset = load_dataset("cnn_dailymail", '3.0.0')
    # article_key = 'article'
    # summary_key = 'highlights'
    # name='cnn_dailymail'
    # dataset=dataset['test']
    
    # # dataset=dataset.select(range(10))

    # name='cnn'

    # #call(generator,title,max_gen_len,temperature,top_p)
    # paraphrased_article = generate_paraphrased_articles(generator,dataset, name, max_gen_len,temperature,top_p)
    # Save data
    
    
    dataset = load_dataset("xsum")
    article_key = 'document'
    summary_key = 'summary'
    dataset=dataset['test']


    name='xsum'
    
    dataset=dataset.select(range(2000))


    paraphrased_article = generate_paraphrased_articles(generator,dataset, name, max_gen_len,temperature,top_p)
    
    dataset = load_dataset("argilla/news-summary")
    article_key = 'text'
    summary_key = 'prediction'
    dataset = DatasetDict({
        'train': dataset['test'],
            'test': dataset['train']})
    dataset=dataset['test']

    name='news'

    #dataset=dataset.select(range(2000))
    
    paraphrased_article = generate_paraphrased_articles(generator,dataset, name, max_gen_len,temperature,top_p)
    

    # dataset = load_dataset('reddit_tifu', 'long')
    # article_key = 'documents'
    # summary_key = 'tldr'
    # # 80% train, 20% test + validation
    # train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
    # # Split the 20% test + valid in half test, half valid
    # test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # # gather everyone if you want to have a single DatasetDict
    # dataset = DatasetDict({
    #     'train': train_testvalid['train'],
    #     'test': test_valid['test'],
    #     'validation': test_valid['train']})

    # dataset=dataset['test']
    # name='reddit'

    # # dataset=dataset.select(range(10))
    

    # paraphrased_article = generate_paraphrased_articles(generator,dataset, name, max_gen_len,temperature,top_p)
