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
from tqdm.auto import tqdm
from natsort import natsorted
import random



def prompt_CNN(article):

    prompt="""

For the following article: {} 

Return a summary comprising of 3 sentences. With each sentence in a numbered list format.

For example:

1. First sentence

2. Second sentence

3. Third Sentence

""".format(article)

    return prompt

def prompt(article):

    prompt="""

For the following article: {} 

Return a summary comprising of 1 sentence. With the sentence in a numbered list format.

For example:

1. First sentence

""".format(article)

    return prompt


print("-"*20)


def get_paraphrased(articles, dataset_name, api_key):
    
    if dataset_name=='cnn_dailymail':
        user_input = prompt_CNN(articles)

    else:
        user_input = prompt(articles)

    
    # return changed sentences in list target sentences
    
    openai.api_key = api_key
    

    while True:
        try:
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": 'You are a helpful assistant that is an expert in summarizing articles.'},
                                    {"role": "user", "content": """
                                                                  
{}
                           
                                    """.format(user_input)}
                        ])

            
            res=response["choices"][0]["message"]["content"]
            # print(res)
            
            sleep(5)
            break
        except Exception as e:
            sleep(5)
            print(e)
            if "This model's maximum context length is 4097 tokens." in str(e):
                return " "
                
                
    


    return res



# def generate_paraphrased_articles(dataset, dataset_name, api_key,batch_size=1):
    
    
#     paraphrase = get_paraphrased(batch_sentences,api_key)


#     # pp_articles.extend(batch_pp_articles)
#     # with open(f'saved_data_new/{dataset_name}/{i}.pkl', 'wb') as f:
#     #     pkl.dump(batch_pp_articles, f)
    
#     return pp_articles


if __name__=='__main__':
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # dataset = load_dataset("cnn_dailymail", '3.0.0')
    # article_key = 'article'
    # summary_key = 'highlights'
    # name='cnn_dailymail'
    # dataset=dataset['test']
    
    # # dataset=dataset.select(range(10))

    # name='cnn'

    
    # # Save data
    
    
    # dataset = load_dataset("xsum")
    # article_key = 'document'
    # summary_key = 'summary'
    # dataset=dataset['test']


    # name='xsum'
    
    # # dataset=dataset.select(range(10))

    

    # paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)
    
    # dataset = load_dataset("argilla/news-summary")
    # article_key = 'text'
    # summary_key = 'prediction'
    # dataset = DatasetDict({
    #     'train': dataset['test'],
    #         'test': dataset['train']})
    # dataset=dataset['test']

    # name='news'
    
    # paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)

    # # dataset=dataset.select(range(10))
    

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

    # name='reddit'

    # # dataset=dataset.select(range(10))
    

    # paraphrased_article = generate_paraphrased_articles(dataset, name, api_key)
    
    
    
    #######################################################################
    
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    
    pkls=os.listdir('../paraphrased_articles/cnn')
    pkls=natsorted(pkls)
    pkls_list=[]
    count=0
    for pikl in pkls:
        with open('../paraphrased_articles/cnn/{}'.format(pikl),'rb') as f:
            file=pkl.load(f)
        if not file:
            count+=1
            file.append(' ') #no paraphrasing possible
        pkls_list.extend(file)

    # print(count)
    # print(len(pkls_list))
    # print(pkls_list[0])
    
    article_key = 'article'
    summary_key = 'highlights'
    name='cnn_dailymail'
    
    
    
    data=dataset['test']
    data = data.remove_columns(article_key).add_column(article_key, pkls_list).cast(data.features)
    #data[article_key]=pkls_list
    print(data)

    # data=data.select(range(10))
    
    
    collected=os.listdir('data_paraphrase/cnn')
    count=0

    for article in tqdm(data[article_key]):
        cnn=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        elif article==' ':
            with open('data_paraphrase/cnn/{}.pkl'.format(count), 'wb') as f:
                pkl.dump([' '],f)
            count+=1
            continue
      
        
        else:    
            try:    
                cnn.append(get_paraphrased(article, name, api_key))
                with open('data_paraphrase/cnn/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(cnn,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_paraphrase/cnn/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
            
            
            count+=1  




    dataset = load_dataset("xsum")
    
    pkls=os.listdir('../paraphrased_articles/xsum')
    pkls=natsorted(pkls)
    pkls_list=[]
    count=0
    for pikl in pkls:
        with open('../paraphrased_articles/xsum/{}'.format(pikl),'rb') as f:
            file=pkl.load(f)
        if not file:
            count+=1
            file.append(' ') #no paraphrasing possible
        pkls_list.extend(file)

    # print(count)
    # print(len(pkls_list))
    # print(pkls_list[0])
    
    
    
    
    article_key = 'document'
    summary_key = 'summary'
    data=dataset['test']
    data = data.remove_columns(article_key).add_column(article_key, pkls_list).cast(data.features)
    #data[article_key]=pkls_list

    # data=data.select(range(10))
    name='xsum'
    

    collected=os.listdir('data_paraphrase/xsum')
    count=0

    for article in tqdm(data[article_key]):
        xsum=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        elif article==' ':
            with open('data_paraphrase/xsum/{}.pkl'.format(count), 'wb') as f:
                pkl.dump([' '],f)
            count+=1
            continue
        
        elif count==797:
            with open('data_paraphrase/xsum/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([' '],f)
            count+=1
            continue
        
        
        elif count==8018:
            with open('data_paraphrase/xsum/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([' '],f)
            count+=1
            continue
            
        
        else:
            try:    
                xsum.append(get_paraphrased(article, name, api_key))
                with open('data_paraphrase/xsum/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(xsum,f)
            except Exception as e:
                print(e)
                print("out of memory")
                with open('data_paraphrase/xsum/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump([' '],f)
            
            
            count+=1    
            

    dataset = load_dataset("argilla/news-summary")
    
    pkls=os.listdir('../paraphrased_articles/news')
    pkls=natsorted(pkls)
    pkls_list=[]
    count=0
    for pikl in pkls:
        with open('../paraphrased_articles/news/{}'.format(pikl),'rb') as f:
            file=pkl.load(f)
        if not file:
            count+=1
            file.append(' ') #no paraphrasing possible
        pkls_list.extend(file)

    # print(count)
    # print(len(pkls_list))
    # print(pkls_list[0])
    
    
    article_key = 'text'
    summary_key = 'prediction'
    dataset = DatasetDict({
        'train': dataset['test'],
            'test': dataset['train']})
    data=dataset['test']
    data = data.remove_columns(article_key).add_column(article_key, pkls_list).cast(data.features)
    #data[article_key]=pkls_list
    
    
    #dataset=dataset.select(range(10))
    name='argilla/news-summary'



    collected=os.listdir('data_paraphrase/news')
    count=0

    for article in tqdm(data[article_key]):
        news=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        elif article==' ':
            with open('data_paraphrase/news/{}.pkl'.format(count), 'wb') as f:
                pkl.dump([' '],f)
            count+=1
            continue
        
        
        else:    
            try:    
                news.append(get_paraphrased(article, name, api_key))
                with open('data_paraphrase/news/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(news,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_paraphrase/news/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
            
            
            count+=1
    



    dataset = load_dataset('reddit_tifu', 'long')
    
    pkls=os.listdir('../paraphrased_articles/reddit')
    pkls=natsorted(pkls)
    pkls_list=[]
    count=0
    for pikl in pkls:
        with open('../paraphrased_articles/reddit/{}'.format(pikl),'rb') as f:
            file=pkl.load(f)
        if not file:
            count+=1
            file.append(' ') #no paraphrasing possible
        pkls_list.extend(file)

    # print(count)
    # print(len(pkls_list))
    # print(pkls_list[0])
    
    
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


    data=dataset['test']
    data = data.remove_columns(article_key).add_column(article_key, pkls_list).cast(data.features)
    #data[article_key]=pkls_list
    #dataset=dataset.select(range(10))
    name='reddit_tifu'

    collected=os.listdir('data_paraphrase/reddit')
    count=0

    for article in tqdm(data[article_key]):
        reddit=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        elif article==' ':
            with open('data_paraphrase/reddit/{}.pkl'.format(count), 'wb') as f:
                pkl.dump([' '],f)
            count+=1
            continue
        
        else:
            try:    
                reddit.append(get_paraphrased(article, name, api_key))
                with open('data_paraphrase/reddit/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(reddit,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_paraphrase/reddit/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
                 
                
            count+=1

