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
import evaluate



def prompt_CNN(article):

    prompt="""

For the following article: {} 

Return a summary comprising of 1 sentences. With each sentence in a numbered list format.

For example:

1. First sentence

""".format(article)

    return prompt



def get_paraphrased(articles, dataset_name, api_key):
    
    
    user_input = prompt_CNN(articles)



    
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


if __name__=='__main__':
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    dataset = load_dataset("xsum")
    
    pkls=os.listdir('../../paraphrased_articles/xsum')
    pkls=natsorted(pkls)
    pkls_list=[]
    count=0
    for pikl in pkls:
        with open('../../paraphrased_articles/xsum/{}'.format(pikl),'rb') as f:
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

    #dataset=dataset.select(range(10))
    name='xsum'
    
    data=data.select(range(100))
    
    overall=[]
    for article in tqdm(data[article_key]):
        cnn=[]
        context = article
        for i in range(3): 
            cnn.append(get_paraphrased(article, name, api_key))
        overall.append(cnn)
    
    with open('results-paraphrased-xsum.pkl', 'wb') as f:
        pkl.dump(overall,f)
    
    overall_rouge=[]

    rouge = evaluate.load('rouge')
    for subset in overall:
        for i in range(len(subset)):
            for j in range(i+1,len(subset)):
                print(i,j)    
                results = rouge.compute(predictions=[subset[i]], references=[subset[j]])
                overall_rouge.append(results['rouge1'])
    
    res=sum(overall_rouge)/len(overall_rouge)
    print(res)
    