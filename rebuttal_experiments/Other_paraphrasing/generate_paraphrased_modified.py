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
import pickle
import evaluate
# from styleformer import Styleformer
import warnings
warnings.filterwarnings("ignore")
# import nlpaug
# import nlpaug.augmenter.word as naw
import copy
import multiprocessing


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


def generate_n_segments(a, n=10): #NEW
  k, m = divmod(len(a), n)
  return list((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))

def parse_args():
    parser = argparse.ArgumentParser(description='Paraphrasing Bias Experiments')
    parser.add_argument('--dataset', type=str, default="cnn", help="cnn/xsum/reddit/news")
    parser.add_argument('--seed', type=int, default=42, help="seed for experiments")
    parser.add_argument('--style', type=str, default='fc', help='deep1/deep2')
    parser.add_argument('--device', type=int, default=0, help="Choose which GPU to run.")
    parser.add_argument('--split', type=str, default='train', help='train/test/validation')
    parser.add_argument('--batch-size', type=int, default=8, help="batch size.")
    args = parser.parse_args()
    return args

args = parse_args()

# Loading Datasets
print("-"*20)
print("Loading datasets....")
if args.dataset == 'cnn':      #USE
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    article_key = 'article'
    summary_key = 'highlights'

elif args.dataset == 'xsum':   #USE
    dataset = load_dataset("xsum")
    article_key = 'document'
    summary_key = 'summary'

elif args.dataset == 'samsum':
    dataset = load_dataset("samsum")
    article_key = 'dialogue'
    summary_key = 'summary'

elif args.dataset == 'news':   #USE
    dataset = load_dataset("argilla/news-summary")
    article_key = 'text'
    summary_key = 'prediction'
    dataset = DatasetDict({
        'train': dataset['test'],
        'test': dataset['train']})
    
elif args.dataset == 'reddit':   #USE
    dataset = load_dataset('reddit_tifu', 'long')
    article_key = 'documents'
    summary_key = 'tldr'
    # 80% train, 20% test + validation
    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=args.seed)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=args.seed)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
else:
    raise Exception('Invalid dataset/Dataset not found.')

dataset=dataset['test']
dataset=dataset.select(range(10))

# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_length = len(dataset['train'][article_key])
# test_length = len(dataset['test'][article_key])
# try:
#     val_length = len(dataset['validation'][article_key])
# except:
#     val_length = 'NA'
# print(f"Dataset stats: {train_length=} | {test_length=} | {val_length=}")

print("-"*20)
print("Loading styling model....")
if args.style == 'deep2':
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
# elif args.style == "fc":
    # model_name = "prithivida/formal_to_informal_styletransfer"
elif args.style == 'deep1':
    model_name = "tuner007/pegasus_paraphrase"
else:
    raise Exception('Invalid conversion style.')

pipe = pipeline(model = model_name, device=args.device)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.config.max_length = 256

def get_paraphrased(input_text):
    target_sentences = []
    
    if args.style=="deep1":
        context = ["transfer Active to Passive: "+ text for text in input_text]

    elif args.style=="deep2":
        context = ["paraphrase: "+ text + " </s>" for text in input_text]

    encoding = tokenizer(context,max_length=model.config.max_length, padding=True, return_tensors="pt")
    input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    model.eval()
    generations = model.generate(
        input_ids=input_ids,attention_mask=attention_mask,
        max_length=model.config.max_length,
        early_stopping=True,
        num_beams=5,
        num_beam_groups = 1,
        num_return_sequences=1,
  )
    for beam_output in generations:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      target_sentences.append(sent)

    return target_sentences

def tokenize(example):
    example["tokenized_document"] = nltk.sent_tokenize(example[article_key])
    if args.dataset == 'news':
        example["tokenized_summary"] = nltk.sent_tokenize(example[summary_key][0][article_key])
    else:
        example["tokenized_summary"] = nltk.sent_tokenize(example[summary_key])
    return example

def generate_paraphrased_articles(dataset, batch_size=args.batch_size):
    dataset = dataset.map(tokenize, num_proc=multiprocessing.cpu_count())
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

        for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):

            idx = get_summary_indices(article, summ, top_k=2, tolerance=0.1)
            sentences = [article[x] for x in idx]
        

            batch_idx.extend(list(idx))
            batch_idx.append(separator)
            
            batch_sentences.extend(sentences) 
            batch_sentences.append(separator)           

        paraphrase = get_paraphrased(batch_sentences)
        pp_articles.append(paraphrase)
        
        # index = 0

        # for j, (article, summ) in enumerate(zip(batch_articles, batch_highlights)):
        #     paraphrased_article = article
        #     while index < len(batch_idx) and batch_idx[index] != separator and batch_idx[index] != 'A':
        #         try:
        #             paraphrased_article[batch_idx[index]] = paraphrase[index]
        #         except:
        #             print("In except.")
        #             print(paraphrased_article)
        #             print(index)
        #             print(batch_idx[index])
        #         index += 1
        #     index += 1
            
        #     batch_pp_articles.append(' '.join(paraphrased_article))

        # pp_articles.extend(batch_pp_articles)
    return pp_articles

import time

start = time.time()
#Paraphrase articles
paraphrased_article = generate_paraphrased_articles(dataset)
print('Execution time: ', time.time()-start)
# Save data
with open(f'saved_data/{args.dataset}/{args.style}_{args.dataset}_{args.split}_{args.seed}.pkl', 'wb') as f:
   pickle.dump(paraphrased_article, f)
