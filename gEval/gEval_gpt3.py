import datasets
from openai import OpenAI
from tqdm.auto import tqdm
from natsort import natsorted
import random
import pickle
import os


############################################
##### Prepare Datasets for Evaluation ######
############################################

# Dataset names
cnn_ds = datasets.load_dataset("cnn_dailymail", "3.0.0")
cnn_ds = cnn_ds['test']
### For 10% sample

random.seed(42)
ten_percent=int(len(cnn_ds)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(cnn_ds)), ten_percent)
random_indices.sort()
# random_indices
cnn_ds=cnn_ds.select(random_indices)

xsum_ds = datasets.load_dataset("xsum")    
xsum_ds = xsum_ds['test']

random.seed(42)
ten_percent=int(len(xsum_ds)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(xsum_ds)), ten_percent)
random_indices.sort()
# random_indices
xsum_ds=xsum_ds.select(random_indices)

news_ds = datasets.load_dataset("argilla/news-summary")
news_ds = news_ds['train']

random.seed(42)
ten_percent=int(len(news_ds)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(news_ds)), ten_percent)
random_indices.sort()
# random_indices
news_ds=news_ds.select(random_indices)

reddit_ds = datasets.load_dataset("reddit_tifu", "long")
train_testvalid = reddit_ds['train'].train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
reddit_ds = test_valid['test']

random.seed(42)
ten_percent=int(len(reddit_ds)*0.115)
# print(ten_percent)
random_indices = random.sample(range(len(reddit_ds)), ten_percent)
random_indices.sort()
# random_indices
reddit_ds=reddit_ds.select(random_indices)


# Article and summary keys
ds_keys = {'cnn_ds':['article', 'highlights'],\
            'xsum_ds':['document', 'summary'], \
            'news_ds':['text', 'prediction'], \
            'reddit_ds':['documents', 'tldr']}


def read_paraphrased_summaries(path):
    """Reads all .pkl files within a directory into strings.

    Args:
        path (str): The path to the directory containing .pkl files.

    Yields:
        tuple: A tuple containing (filename, file_content_string) for each .pkl file.
    """

    out = []
    files = natsorted(os.listdir(path))
    for filename in files:
        if filename.endswith(".pkl"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'rb') as f:
                file_content = pickle.load(f)
                out.append(file_content[0])
    return out



def read_summaries(path):
    out = []
    with open(path, 'rb') as f:
        content = pickle.load(f)
        for ls in content:
            out.append('\n'.join(ls))
    return out


origSummaryPath = '../mistral-7b/data_NAACL/run1/'
cnn_sum_orig = read_summaries(origSummaryPath+'cnn.pkl')
xsum_sum_orig = read_summaries(origSummaryPath+'xsum.pkl')
news_sum_orig = read_summaries(origSummaryPath+'news.pkl')
reddit_sum_orig = read_summaries(origSummaryPath+'reddit.pkl')


paraSummaryPath = '../mistral-7b/data_NAACL/run2/'
cnn_sum_para = read_summaries(paraSummaryPath+'cnn.pkl')
xsum_sum_para = read_summaries(paraSummaryPath+'xsum.pkl')
news_sum_para = read_summaries(paraSummaryPath+'news.pkl')
reddit_sum_para = read_summaries(paraSummaryPath+'reddit.pkl')


###############################################
##### ChatGPT-3.5-T Evaluation Functions ###### 
###############################################

def ratePromptGood(i,article, summary):
    prompt = """{i}

    You will be given one summary written for a news article. Your task is to rate the summary based on the following criteria:
    Output format: PERCENTAGE, PERCENTAGE, PERCENTAGE, PERCENTAGE, PERCENTAGE

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it resents them in a clear and logical order.
    3. Rate the summary with 5 percentages, where each one represents how likely the summary is going to get a score from 1 to 5. For example, if you think the summary is 80% likely to get a score of 5, 10% likely to get a score of 4, 5% likely to get a score of 3, 3% likely to get a score of 2, and 1% likely to get a score of 1, you should rate the summary as 80, 10, 5, 3, 2.

    Here is the article: {article}

    Here is the summary: {summary}
    """.format(i=i,article=article, summary=summary)

    return prompt

## Create Prompts
def createPrompts(ds, ds_key, sum_orig, sum_para):
    articleKey = ds_keys[ds_key][0]
    if ds_key!=news_ds:
        summaryKey = ds_keys[ds_key][1]
    
    # Only select 10% of the articles to rate
    # random.seed(42)
    # list_size = len(ds)
    # num_true = int(list_size * 0.115)
    # skip = [False] * num_true + [True] * (list_size - num_true)
    # random.shuffle(skip)

    idx = []
    prompts = []
    for i in range(len(ds)):
        # if skip[i]:
        #     continue

        if (sum_para[i].strip() == '') or (sum_orig[i].strip() == ''):
            continue

        idx.append(str(i))
        article = ds[i][articleKey]
        if ds_key!=news_ds:
            summary = ds[i][summaryKey]
        else:
            summary = ds[i]['prediction']['text']
            
        prompt = ratePromptGood(i,article, summary)
        prompts.append(prompt)
        prompt = ratePromptGood(i,article, sum_orig[i])
        prompts.append(prompt)
        prompt = ratePromptGood(i,article, sum_para[i])
        prompts.append(prompt)

    return idx, prompts


cnn_idx, cnn_prompts = createPrompts(cnn_ds, 'cnn_ds', cnn_sum_orig, cnn_sum_para)
# print(cnn_prompts[0:3])
with open('mistral_cnn_idx.txt', 'w') as f:
    f.write('\n'.join(cnn_idx))
with open('mistral_cnn_prompts.txt', 'w') as f:
    f.write('\n'.join(cnn_prompts))
xsum_idx, xsum_prompts = createPrompts(xsum_ds, 'xsum_ds', xsum_sum_orig, xsum_sum_para)
with open('mistral_xsum_idx.txt', 'w') as f:
    f.write('\n'.join(xsum_idx))
with open('mistral_xsum_prompts.txt', 'w') as f:
    f.write('\n'.join(xsum_prompts))
# print(xsum_prompts[0:3])
news_idx, news_prompts = createPrompts(news_ds, 'news_ds', news_sum_orig, news_sum_para)
with open('mistral_news_idx.txt', 'w') as f:
    f.write('\n'.join(news_idx))
with open('mistral_news_prompts.txt', 'w') as f:
    f.write('\n'.join(news_prompts))
# print(news_prompts[0:3])
reddit_idx, reddit_prompts = createPrompts(reddit_ds, 'reddit_ds', reddit_sum_orig, reddit_sum_para)
with open('mistral_reddit_idx.txt', 'w') as f:
    f.write('\n'.join(reddit_idx))
with open('mistral_reddit_prompts.txt', 'w') as f:
    f.write('\n'.join(reddit_prompts))
print(reddit_prompts[0:3])

## Create ChatGPT client
def ask_chatgpt(idx, prompts, outFilePath):
    """Queries ChatGPT-3.5-turbo with a list of prompts and returns the responses.

    Args:
        prompts (list): A list of strings representing the prompts.
        outFilePath (str): The path to the file where the responses will be written.
    """
    client = OpenAI(
        # This is the default and can be omitted
        api_key="sk-proj-ytd0znrkPJrGvJTReBagT3BlbkFJSmKmf2v9too5HhoZX7m7" #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"
    )
    
    print(len(prompts))

    for i in tqdm(range(len(idx))):
        prompt = prompts[i]
        ans = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0,
        )
        ans = ans.choices[0].message.content
        ans = str(idx[int(i/2)]) + '\t' + ans

        with open(outFilePath, 'a') as f:
            f.write(ans+'\n')

ask_chatgpt(cnn_idx, cnn_prompts, 'mistral_cnn_responses.txt')
ask_chatgpt(xsum_idx, xsum_prompts, 'mistral_xsum_responses.txt')
ask_chatgpt(news_idx, news_prompts, 'mistral_news_responses.txt')
ask_chatgpt(reddit_idx, reddit_prompts, 'mistral_reddit_responses.txt')