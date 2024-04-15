# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

# from llama import Llama, Dialog

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
# from dotenv import load_dotenv
import os
from time import sleep
from evaluate import load


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


# def generate_n_segments(a, n=10): #NEW
#   k, m = divmod(len(a), n)
#   return list((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))


# def get_paraphrased(generator, batch_sentences, max_gen_len,temperature,top_p):
#     target_sentences = []
    
#     for sent in batch_sentences:
#         dialogs: List[Dialog] = []

#         user_input = prompt(sent)

#         dialogs.append([{"role": "user", "content": "{}".format(user_input)}])
            
    

        
#         results = generator.chat_completion(
#             dialogs,  # type: ignore
#             max_gen_len=max_gen_len,
#             temperature=temperature,
#             top_p=top_p,
#         )


#         for result in results:

            
#             final=result['generation']['content']

#         try:
#             final=final.split(':')[1].split('\n\n')[1]
#             print(final)
#             target_sentences.append(final)
#         except Exception as e:
#             print(e)
#             print(final)
#             return "-1"
        
#     return target_sentences

def get_bertscore(original_sentences_final,paraphrased_sentences_final):
    
    highlights = []
    model_s = []

    for j,k in zip(original_sentences_final,paraphrased_sentences_final):
        if not j or not k:
            print('in skip')
            continue
        else:
            highlights.append(' '.join(j))
            model_s.append(' '.join(k))
    
    bertscore = load("bertscore")
    
    results = bertscore.compute(predictions=model_s, references=highlights, lang="en", device='cuda:1')
    mean_precision=sum(results['precision'])/len(results['precision'])
    mean_recall=sum(results['recall'])/len(results['recall'])
    mean_f1=sum(results['f1'])/len(results['f1'])
    
    return mean_precision,mean_recall,mean_f1
        
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


def get_original_sentences(dataset, dataset_name,batch_size=1):
    
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
        # collected=os.listdir(f"../paraphrased_articles/{dataset_name}")
        
        # if "{}.pkl".format(i) in collected:
        #     continue

        for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):
            
            flag=False
          
            try:
                idx = get_summary_indices(article, summ, top_k=2, tolerance=0.1)
                # print(idx)
            except Exception as e:
                #print("in Except")
                #print(e)
                pp_articles.append([])
                break
                
                
                
                
            original_sentences = [article[x] for x in idx]
            # if (idx != 'A').all():
            #     sentences = [article[x] for x in idx]
            # else:
            #     sentences = ['No']

            batch_idx.extend(list(idx))
            batch_idx.append(separator)
            
            batch_sentences.extend(original_sentences) 
            pp_articles.append(batch_sentences)

    
    return pp_articles



def get_paraphrased_sentences(dataset, dataset_name,batch_size=1):
    paraphrased_articles=[]
    for i in tqdm(range(len(dataset))):
        with open('../../paraphrased_articles/{}/{}.pkl'.format(dataset_name,i), 'rb') as f:
            untokenized_article=pkl.load(f)
            paraphrased_articles.append(" ".join(untokenized_article))
    
    dataset = dataset.remove_columns(article_key)
    dataset = dataset.add_column(article_key, paraphrased_articles)
    
    # print(dataset[article_key][0])
    
    paraphrased_sent=get_original_sentences(dataset,dataset_name)
    
    return paraphrased_sent
        
        
            

    
 
    

if __name__ == "__main__":


    
    # dataset = load_dataset("cnn_dailymail", '3.0.0')
    # article_key = 'article'
    # summary_key = 'highlights'
    # name='cnn_dailymail'
    # dataset=dataset['test']
    
    # # dataset=dataset.select(range(1000))
    # # first_unpara=dataset[article_key][0]
    # name='cnn'
    
    # dataset_original = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    # dataset_original_articles=dataset_original["tokenized_document"]
    # original_sentences = get_original_sentences(dataset, name)
    
    # paraphrased_articles=[]
    # for i in tqdm(range(len(dataset))):
    #     with open('../../paraphrased_articles/{}/{}.pkl'.format('cnn',i), 'rb') as f:
    #         untokenized_article=pkl.load(f)
    #         paraphrased_articles.append(" ".join(untokenized_article))
    
    # dataset = dataset.remove_columns(article_key)
    # dataset = dataset.add_column(article_key, paraphrased_articles)
    # # first_para=dataset[article_key][0]
    # # print(get_bertscore(first_unpara,first_para))
    # # dataset=dataset.select(range(1000))
    # # print(dataset)
    # # print(dataset['highlights'])
    
    # dataset_paraphrased = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    # dataset_paraphrased_articles=dataset_paraphrased["tokenized_document"]
    
    # # print(dataset_original_articles)
    # # print(dataset_paraphrased_articles)
    

    # #call(generator,title,max_gen_len,temperature,top_p)
    
    # # Save data
    
    # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # # print(len(original_sentences_cnn))
    # # print(len(paraphrased_sentences_cnn))
    
    # original_sentences_final=[]
    # paraphrased_sentences_final=[]
    
    # for orig, para in zip(original_sentences,paraphrased_sentences):
    #     if para:
    #         original_sentences_final.append(orig)
    #         paraphrased_sentences_final.append(para)
    #     else:
    #         #print(para)
    #         #print('empty')
    #         pass
    
    # print(len(original_sentences_final))
    # print(len(paraphrased_sentences_final))
    
    
    # # print(original_sentences_final[0:2])
    # # print(paraphrased_sentences_final[0:2])
    
    # # highlights = []
    # # model_s = []
    
    # # for j in original_sentences_final:
    # #     highlights.append(' '.join(j))

    # # for k in paraphrased_sentences_final:
    # #     model_s.append(' '.join(k))


    # # rouge = evaluate.load('rouge')

    # # #print("==> Comparing generated summaries with gold summaries")
    # # results = rouge.compute(predictions=model_s, references=highlights)
    # # print(results)
    # print("bertscore")
    # print(get_bertscore(original_sentences_final,paraphrased_sentences_final))
    
    
    
    # dataset = load_dataset("xsum")
    # article_key = 'document'
    # summary_key = 'summary'
    # dataset=dataset['test']


    # name='xsum'
    
    # dataset_original = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    # dataset_original_articles=dataset_original["tokenized_document"]
    # original_sentences = get_original_sentences(dataset, name)
    
    # paraphrased_articles=[]
    # for i in tqdm(range(len(dataset))):
    #     with open('../../paraphrased_articles/{}/{}.pkl'.format('xsum',i), 'rb') as f:
    #         untokenized_article=pkl.load(f)
    #         paraphrased_articles.append(" ".join(untokenized_article))
    
    # dataset = dataset.remove_columns(article_key)
    # dataset = dataset.add_column(article_key, paraphrased_articles)
    # # first_para=dataset[article_key][0]
    # # print(get_bertscore(first_unpara,first_para))
    # # dataset=dataset.select(range(1000))
    # # print(dataset)
    # # print(dataset['highlights'])
    
    # dataset_paraphrased = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    # dataset_paraphrased_articles=dataset_paraphrased["tokenized_document"]
    
    # # print(dataset_original_articles)
    # # print(dataset_paraphrased_articles)
    

    # #call(generator,title,max_gen_len,temperature,top_p)
    
    # # Save data
    
    # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # # dataset=dataset.select(range(10))


    # #original_sentences_xsum = get_original_sentences(dataset, name)
    
    #     #call(generator,title,max_gen_len,temperature,top_p)
    # # original_sentences = get_original_sentences(dataset, name)
    # # # Save data
    
    # # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # # print(len(original_sentences_cnn))
    # # print(len(paraphrased_sentences_cnn))
    
    # original_sentences_final=[]
    # paraphrased_sentences_final=[]
    
    # for orig, para in zip(original_sentences,paraphrased_sentences):
    #     if para:
    #         original_sentences_final.append(orig)
    #         paraphrased_sentences_final.append(para)
    #     else:
    #         # print(para)
    #         # print('empty')
    #         pass
    
    # # print(len(original_sentences_cnn_final))
    # # print(len(paraphrased_sentences_cnn_final))
    
    
    # # # original_sentences_cnn_final[0:20]
    # # # paraphrased_sentences_cnn_final[0:20]
    
    # # highlights = []
    # # model_s = []
    
    # # for j in original_sentences_final:
    # #     highlights.append(' '.join(j))

    # # for k in paraphrased_sentences_final:
    # #     model_s.append(' '.join(k))


    # # rouge = evaluate.load('rouge')

    # # #print("==> Comparing generated summaries with gold summaries")
    # # results = rouge.compute(predictions=model_s, references=highlights)
    # #print(results)
    # print("bertscore")
    # print(get_bertscore(original_sentences_final,paraphrased_sentences_final))
    
    
    # dataset = load_dataset("argilla/news-summary")
    # article_key = 'text'
    # summary_key = 'prediction'
    # dataset = DatasetDict({
    #     'train': dataset['test'],
    #         'test': dataset['train']})
    # dataset=dataset['test']

    # name='news'

    # dataset_original = dataset.map(tokenize_news, num_proc=multiprocessing.cpu_count())
    # dataset_original_articles=dataset_original["tokenized_document"]
    # original_sentences = get_original_sentences(dataset, name)
    
    # paraphrased_articles=[]
    # for i in tqdm(range(len(dataset))):
    #     with open('../../paraphrased_articles/{}/{}.pkl'.format('news',i), 'rb') as f:
    #         untokenized_article=pkl.load(f)
    #         paraphrased_articles.append(" ".join(untokenized_article))
    
    # dataset = dataset.remove_columns(article_key)
    # dataset = dataset.add_column(article_key, paraphrased_articles)
    # # first_para=dataset[article_key][0]
    # # print(get_bertscore(first_unpara,first_para))
    # # dataset=dataset.select(range(1000))
    # # print(dataset)
    # # print(dataset['highlights'])
    
    # dataset_paraphrased = dataset.map(tokenize_news, num_proc=multiprocessing.cpu_count())
    # dataset_paraphrased_articles=dataset_paraphrased["tokenized_document"]
    
    # # print(dataset_original_articles)
    # # print(dataset_paraphrased_articles)
    

    # #call(generator,title,max_gen_len,temperature,top_p)
    
    # # Save data
    
    # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # # dataset=dataset.select(range(10))


    # #original_sentences_xsum = get_original_sentences(dataset, name)
    
    #     #call(generator,title,max_gen_len,temperature,top_p)
    # # original_sentences = get_original_sentences(dataset, name)
    # # # Save data
    
    # # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # # print(len(original_sentences_cnn))
    # # print(len(paraphrased_sentences_cnn))
    
    # original_sentences_final=[]
    # paraphrased_sentences_final=[]
    
    # for orig, para in zip(original_sentences,paraphrased_sentences):
    #     if para:
    #         original_sentences_final.append(orig)
    #         paraphrased_sentences_final.append(para)
    #     else:
    #         # print(para)
    #         # print('empty')
    #         pass
    
    # # print(len(original_sentences_cnn_final))
    # # print(len(paraphrased_sentences_cnn_final))
    
    
    # # # original_sentences_cnn_final[0:20]
    # # # paraphrased_sentences_cnn_final[0:20]
    
    # # highlights = []
    # # model_s = []
    
    # # for j in original_sentences_final:
    # #     highlights.append(' '.join(j))

    # # for k in paraphrased_sentences_final:
    # #     model_s.append(' '.join(k))


    # # rouge = evaluate.load('rouge')

    # # #print("==> Comparing generated summaries with gold summaries")
    # # results = rouge.compute(predictions=model_s, references=highlights)
    # #print(results)
    # print("bertscore")
    # print(get_bertscore(original_sentences_final,paraphrased_sentences_final))
    

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

    dataset=dataset['test']
    name='reddit'
    dataset=dataset.select(range(1000))
  
    
    dataset_original = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    dataset_original_articles=dataset_original["tokenized_document"]
    original_sentences = get_original_sentences(dataset, name)
    
    paraphrased_articles=[]
    for i in tqdm(range(len(dataset))):
        with open('../../paraphrased_articles/{}/{}.pkl'.format('reddit',i), 'rb') as f:
            untokenized_article=pkl.load(f)
            paraphrased_articles.append(" ".join(untokenized_article))
    
    dataset = dataset.remove_columns(article_key)
    dataset = dataset.add_column(article_key, paraphrased_articles)
    # first_para=dataset[article_key][0]
    # print(get_bertscore(first_unpara,first_para))
    dataset=dataset.select(range(1000))
    # print(dataset)
    # print(dataset['highlights'])
    
    dataset_paraphrased = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
    dataset_paraphrased_articles=dataset_paraphrased["tokenized_document"]
    
    # print(dataset_original_articles)
    # print(dataset_paraphrased_articles)
    

    #call(generator,title,max_gen_len,temperature,top_p)
    
    # Save data
    
    paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # dataset=dataset.select(range(10))


    #original_sentences_xsum = get_original_sentences(dataset, name)
    
        #call(generator,title,max_gen_len,temperature,top_p)
    # original_sentences = get_original_sentences(dataset, name)
    # # Save data
    
    # paraphrased_sentences = get_paraphrased_sentences(dataset, name)
    
    # print(len(original_sentences_cnn))
    # print(len(paraphrased_sentences_cnn))
    
    original_sentences_final=[]
    paraphrased_sentences_final=[]
    
    for orig, para in zip(original_sentences,paraphrased_sentences):
        if para:
            original_sentences_final.append(orig)
            paraphrased_sentences_final.append(para)
        else:
            # print(para)
            # print('empty')
            pass
    
    # print(len(original_sentences_cnn_final))
    # print(len(paraphrased_sentences_cnn_final))
    
    
    # # original_sentences_cnn_final[0:20]
    # # paraphrased_sentences_cnn_final[0:20]
    
    # highlights = []
    # model_s = []
    
    # for j in original_sentences_final:
    #     highlights.append(' '.join(j))

    # for k in paraphrased_sentences_final:
    #     model_s.append(' '.join(k))


    # rouge = evaluate.load('rouge')

    # #print("==> Comparing generated summaries with gold summaries")
    # results = rouge.compute(predictions=model_s, references=highlights)
    #print(results)
    sanity_check={'Original':original_sentences_final, 'Paraphrased':paraphrased_sentences_final}
    df=pd.DataFrame.from_dict(sanity_check)
    df.to_csv('Reddit.csv', index=False)
    print("bertscore")
    print(get_bertscore(original_sentences_final,paraphrased_sentences_final))
    