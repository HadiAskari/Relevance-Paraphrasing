import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from datasets import load_dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
import pickle as pkl
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import transformers
import random

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


model_name = "databricks/dolly-v2-7b"
#prompt = "Tell me about gravity"
#access_token = "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"



model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

generate_text = pipeline(model="databricks/dolly-v2-7b", trust_remote_code=True, device_map="auto",return_full_text=False, do_sample=False,
        max_new_tokens=500, 
        temperature=0.7)


data = load_dataset("cnn_dailymail", '3.0.0')
article_key = 'article'
summary_key = 'highlights'
data=data['test']

### For 10% sample

random.seed(42)
ten_percent=int(len(data)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(data)), ten_percent)
random_indices.sort()
# random_indices


data=data.select(random_indices)
    

# template for an instruction with input
# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nArticle:\n{context}")

# hf_pipeline = HuggingFacePipeline(pipeline=generate_text)


# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)


collected=os.listdir('data_NAACL/run2/cnn')
count=0

for article in tqdm(data[article_key]):
    cnn=[]
    context = article
    
    if '{}.pkl'.format(count) in collected:
        count+=1
        continue
    
    elif count==8018:
        with open('data_NAACL/run2/cnn/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(cnn,f)
        count+=1
        
    elif article==' ':
        with open('data_NAACL/run2/cnn/{}.pkl'.format(count), 'wb') as f:
            pkl.dump([' '],f)
        count+=1
        continue
    else:    
        cnn.append(generate_text("Generate a 3 sentence summary for the given article in a numbered list format. Here is the article: \n Article: {}. \n Summary: ".format(context)))
        with open('data_NAACL/run2/cnn/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(cnn,f)
        count+=1

data = load_dataset("xsum")
article_key = 'document'
summary_key = 'summary'
data=data['test']
### For 10% sample

random.seed(42)
ten_percent=int(len(data)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(data)), ten_percent)
random_indices.sort()
# random_indices


data=data.select(random_indices)
    

# # template for an instruction with input
# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nArticle:\n{context}")

# hf_pipeline = HuggingFacePipeline(pipeline=generate_text)


# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

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
            pkl.dump(xsum,f)
        count+=1
    
    elif article==' ':
        with open('data_NAACL/run2/xsum/{}.pkl'.format(count), 'wb') as f:
            pkl.dump([' '],f)
        count+=1
        continue
    
    else:    
        xsum.append(generate_text("Generate a 1 sentence summary for the given article in a numbered list format. Here is the article: \n Article: {}. \n Summary: ".format(context)))
        with open('data_NAACL/run2/xsum/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(xsum,f)
        count+=1



data = load_dataset("argilla/news-summary")
article_key = 'text'
summary_key = 'prediction'
data = DatasetDict({
    'train': data['test'],
    'test': data['train']})

data=data['test']

### For 10% sample

random.seed(42)
ten_percent=int(len(data)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(data)), ten_percent)
random_indices.sort()
# random_indices


data=data.select(random_indices)
    

# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nArticle:\n{context}")

# hf_pipeline = HuggingFacePipeline(pipeline=generate_text)


# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

collected=os.listdir('data_NAACL/run2/news')
count=0

for article in tqdm(data[article_key]):
    news=[]
    context = article
    
    if '{}.pkl'.format(count) in collected:
        count+=1
        continue
    
    elif count==8018:
        with open('data_NAACL/run2/news/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(news,f)
        count+=1
        
    elif article==' ':
        with open('data_NAACL/run2/news/{}.pkl'.format(count), 'wb') as f:
            pkl.dump([' '],f)
        count+=1
        continue
    else:    
        news.append(generate_text("Generate a 1 sentence summary for the given article in a numbered list format. Here is the article: \n Article: {}. \n Summary: ".format(context)))
        with open('data_NAACL/run2/news/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(news,f)
        count+=1



data = load_dataset('reddit_tifu', 'long')
article_key = 'documents'
summary_key = 'tldr'
    # 80% train, 20% test + validation
train_testvalid = data['train'].train_test_split(test_size=0.2, seed=42)
# Split the 20% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
# gather everyone if you want to have a single DatasetDict
data = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

data=data['test']

### For 10% sample

random.seed(42)
ten_percent=int(len(data)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(data)), ten_percent)
random_indices.sort()
# random_indices


data=data.select(random_indices)
    

# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nArticle:\n{context}")

# hf_pipeline = HuggingFacePipeline(pipeline=generate_text)


# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

collected=os.listdir('data_NAACL/run2/reddit')
count=0

for article in tqdm(data[article_key]):
    reddit=[]
    context = article
    
    if '{}.pkl'.format(count) in collected:
        count+=1
        continue
    
    elif count==8018:
        with open('data_NAACL/run2/reddit/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(reddit,f)
        count+=1
    elif article==' ':
        with open('data_NAACL/run2/reddit/{}.pkl'.format(count), 'wb') as f:
            pkl.dump([' '],f)
        count+=1
        continue
    else:    
        reddit.append(generate_text("Generate a 1 sentence summary for the given article in a numbered list format. Here is the article: \n Article: {}. \n Summary: ".format(context)))
        with open('data_NAACL/run2/reddit/{}.pkl'.format(count), 'wb') as f:
            pkl.dump(reddit,f)
        count+=1