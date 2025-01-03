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
import random

import os


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



def build(ckpt_dir,tokenizer_path,max_seq_len,max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

# def prompt_output_parse_CNN(output):
#     first_split=output.split('\n\n')
#     sentences=[]
#     count=0
#     for items in first_split:
#         if count==3:
#             break
#         split=items.split('.')[0].strip()
#         if split:
#             sentences.append(items.split(':')[-1].strip())
#             count+=1
        

#     return sentences

# def prompt_output_parse(output):
#     first_split=output.split('\n\n')
#     sentences=[]
#     count=0
#     for items in first_split:
#         if count==1:
#             break
#         split=items.split('.')[0].strip()
#         if split:
#             sentences.append(items.split(':')[-1].strip())
#             count+=1
        

#     return sentences

def call(generator, articles, name, max_gen_len,temperature,top_p):
    #print(len(article))
    dialogs: List[Dialog] = []

    if name=='cnn_dailymail':
        user_input = prompt_CNN(articles)

    else:
        user_input = prompt(articles)

    dialogs.append([{"role":"user","content":"From now on you are an expert summarizer of articles that will help me generate summaries."},{"role":"assistant", "content": "Sure! I can help you in generating summaries for articles. Please go ahead and send me the article."},{"role": "user", "content": "{}".format(user_input)}])
        
    #print(len(dialogs))

    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    #print(results)
    for result in results:
        # for msg in dialog:
            # print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        # if name=='cnn_dailymail':
        #     final=prompt_output_parse_CNN(result['generation']['content'])

        # else:
        #     final=prompt_output_parse(result['generation']['content'])
        
        final=result['generation']['content']
    
    #print(final)
    return final
        
    

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
):

    generator = build(ckpt_dir,tokenizer_path,max_seq_len,max_batch_size)
    

    
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    article_key = 'article'
    summary_key = 'highlights'
    name='cnn_dailymail'
    data=dataset['test']
    
    ### For 10% sample
    
    random.seed(42)
    ten_percent=int(len(data)*0.115)
    # print(ten_percent)
    random_indices = random.sample(range(len(data)), ten_percent)
    random_indices.sort()
    # random_indices

    
    data=data.select(random_indices)
    
    collected=os.listdir('data_NAACL/run2/cnn')
    count=0

    for article in tqdm(data[article_key]):
        cnn=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        else:    
            try:    
                cnn.append(call(generator,article, name,max_gen_len,temperature,top_p))
                with open('data_NAACL/run2/cnn/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(cnn,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_NAACL/run2/cnn/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
            
            
            count+=1  




    dataset = load_dataset("xsum")
    article_key = 'document'
    summary_key = 'summary'
    data=dataset['test']

    random.seed(42)
    ten_percent=int(len(data)*0.115)
    # print(ten_percent)
    random_indices = random.sample(range(len(data)), ten_percent)
    random_indices.sort()
    # random_indices


    data=data.select(random_indices)
    name='xsum'


    collected=os.listdir('data_NAACL/run2/xsum')
    count=0

    for article in tqdm(data[article_key]):
        xsum=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        elif count==8018:
            with open('data_NAACL/run2/xsum/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([' '],f)
            count+=1
            continue
            
        
        else:
            try:    
                xsum.append(call(generator,article, name,max_gen_len,temperature,top_p))
                with open('data_NAACL/run2/xsum/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(xsum,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_NAACL/run2/xsum/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
            
            
            count+=1    
            

    dataset = load_dataset("argilla/news-summary")
    article_key = 'text'
    summary_key = 'prediction'
    dataset = DatasetDict({
        'train': dataset['test'],
            'test': dataset['train']})
    data=dataset['test']
    #dataset=dataset.select(range(10))
    name='argilla/news-summary'
    
    random.seed(42)
    ten_percent=int(len(data)*0.115)
    # print(ten_percent)
    random_indices = random.sample(range(len(data)), ten_percent)
    random_indices.sort()
    # random_indices


    data=data.select(random_indices)



    collected=os.listdir('data_NAACL/run2/news')
    count=0

    for article in tqdm(data[article_key]):
        news=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        
        else:    
            try:    
                news.append(call(generator,article, name,max_gen_len,temperature,top_p))
                with open('data_NAACL/run2/news/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(news,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_NAACL/run2/news/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
            
            
            count+=1
    



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


    data=dataset['test']
    #dataset=dataset.select(range(10))
    name='reddit_tifu'
    
    random.seed(42)
    ten_percent=int(len(data)*0.115)
    # print(ten_percent)
    random_indices = random.sample(range(len(data)), ten_percent)
    random_indices.sort()
    # random_indices


    data=data.select(random_indices)

    collected=os.listdir('data_NAACL/run2/reddit')
    count=0

    for article in tqdm(data[article_key]):
        reddit=[]
        context = article
        
        if '{}.pkl'.format(count) in collected:
            count+=1
            continue
        
        
        else:
            try:    
                reddit.append(call(generator,article, name,max_gen_len,temperature,top_p))
                with open('data_NAACL/run2/reddit/{}.pkl'.format(count), 'wb') as f:
                    pkl.dump(reddit,f)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("out of memory")
                    with open('data_NAACL/run2/reddit/{}.pkl'.format(count), 'wb') as f:
                        pkl.dump([],f)
                 
                    
                
            count+=1
    




if __name__ == "__main__":
    fire.Fire(main)
