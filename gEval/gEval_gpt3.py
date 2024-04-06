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

xsum_ds = datasets.load_dataset("xsum")    
xsum_ds = xsum_ds['test']

news_ds = datasets.load_dataset("argilla/news-summary")
news_ds = news_ds['train']

reddit_ds = datasets.load_dataset("reddit_tifu", "long")
train_testvalid = reddit_ds['train'].train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
reddit_ds = test_valid['test']


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


origSummaryPath = '../gpt3.5-T/data_original/'
cnn_sum_orig = read_summaries(origSummaryPath+'cnn_new_new.pkl')
xsum_sum_orig = read_summaries(origSummaryPath+'xsum_capped_random.pkl')
news_sum_orig = read_summaries(origSummaryPath+'news_capped_random.pkl')
reddit_sum_orig = read_summaries(origSummaryPath+'reddit_new_new.pkl')


paraSummaryPath = '../gpt3.5-T/data_paraphrase/'
cnn_sum_para = read_summaries(paraSummaryPath+'cnn.pkl')
xsum_sum_para = read_summaries(paraSummaryPath+'xsum.pkl')
news_sum_para = read_summaries(paraSummaryPath+'news.pkl')
reddit_sum_para = read_summaries(paraSummaryPath+'reddit.pkl')


###############################################
##### ChatGPT-3.5-T Evaluation Functions ###### 
###############################################

def ratePromptGood(article, summary):
    prompt = """

    You will be given one summary written for a news article. Your task is to rate the summary based on the following criteria:
    Output format: PERCENTAGE, PERCENTAGE, PERCENTAGE, PERCENTAGE, PERCENTAGE

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it resents them in a clear and logical order.
    3. Rate the summary with 5 percentages, where each one represents how likely the summary is going to get a score from 1 to 5. For example, if you think the summary is 80% likely to get a score of 5, 10% likely to get a score of 4, 5% likely to get a score of 3, 3% likely to get a score of 2, and 1% likely to get a score of 1, you should rate the summary as 80, 10, 5, 3, 2.

    Here is the article: {article}

    Here is the summary: {summary}
    """.format(article=article, summary=summary)

    return prompt

## Create Prompts
def createPrompts(ds, ds_key, sum_orig, sum_para):
    articleKey = ds_keys[ds_key][0]
    summaryKey = ds_keys[ds_key][1]
    # Only select 10% of the articles to rate
    random.seed(42)
    list_size = len(ds)
    num_true = int(list_size * 0.115)
    skip = [False] * num_true + [True] * (list_size - num_true)
    random.shuffle(skip)

    idx = []
    prompts = []
    for i in range(len(ds)):
        if skip[i]:
            continue

        if (sum_para[i].strip() == '') or (sum_orig[i].strip() == ''):
            continue

        idx.append(str(i))
        article = ds[i][articleKey]
        summary = ds[i][summaryKey]
        prompt = ratePromptGood(article, summary)
        prompts.append(prompt)
        prompt = ratePromptGood(article, sum_orig[i])
        prompts.append(prompt)
        prompt = ratePromptGood(article, sum_para[i])
        prompts.append(prompt)

    return idx, prompts


cnn_idx, cnn_prompts = createPrompts(cnn_ds, 'cnn_ds', cnn_sum_orig, cnn_sum_para)
with open('gpt_cnn_idx.txt', 'w') as f:
    f.write('\n'.join(cnn_idx))

xsum_idx, xsum_prompts = createPrompts(xsum_ds, 'xsum_ds', xsum_sum_orig, xsum_sum_para)
with open('gpt_xsum_idx.txt', 'w') as f:
    f.write('\n'.join(xsum_idx))

news_idx, news_prompts = createPrompts(news_ds, 'news_ds', news_sum_orig, news_sum_para)
with open('gpt_news_idx.txt', 'w') as f:
    f.write('\n'.join(news_idx))

reddit_idx, reddit_prompts = createPrompts(reddit_ds, 'reddit_ds', reddit_sum_orig, reddit_sum_para)
with open('gpt_reddit_idx.txt', 'w') as f:
    f.write('\n'.join(reddit_idx))


## Create ChatGPT client
def ask_chatgpt(idx, prompts, outFilePath):
    """Queries ChatGPT-3.5-turbo with a list of prompts and returns the responses.

    Args:
        prompts (list): A list of strings representing the prompts.
        outFilePath (str): The path to the file where the responses will be written.
    """
    client = OpenAI(
        # This is the default and can be omitted
        api_key="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"
    )

    for i in tqdm(range(len(prompts))):
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
        ans = str(idx[int(i/3)]) + '\t' + ans

        with open(outFilePath, 'a') as f:
            f.write(ans+'\n')

ask_chatgpt(cnn_idx, cnn_prompts, 'gpt_cnn_responses.txt')
ask_chatgpt(xsum_idx, xsum_prompts, 'gpt_xsum_responses.txt')
ask_chatgpt(news_idx, news_prompts, 'gpt_news_responses.txt')
ask_chatgpt(reddit_idx, reddit_prompts, 'gpt_reddit_responses.txt')