from icecream import ic
from llama import Llama, Dialog
from typing import List
import sys
import fire
import json


with open("summaries.json", "r") as f:
    summaries = json.load(f)

with open("article.txt", "r") as f:
    article = f.read()


def ratePromptBad(article, summary):
    prompt = """

    You will be given one summary written for a news article. Your task is to rate the summary based on the following criteria:

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it resents them in a clear and logical order.
    3. Rate the summary with 5 percentages, where each one represents how likely the summary is going to get a score from 1 to 5. For example, if you think the summary is 2% likely to get a score of 5, 3% likely to get a score of 4, 5% likely to get a score of 3, 10% likely to get a score of 2, and 80% likely to get a score of 1, you should rate the summary as 2, 3, 5, 10, 80.
    4. Only report back with your final rating. For example: 2, 3, 5, 10, 80

    Here is the article: {article}

    Here is the summary: {summary}
    """.format(article=article, summary=summary)

    return prompt


def ratePromptGood(article, summary):
    prompt = """

    You will be given one summary written for a news article. Your task is to rate the summary based on the following criteria:

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it resents them in a clear and logical order.
    3. Rate the summary with 5 percentages, where each one represents how likely the summary is going to get a score from 1 to 5. For example, if you think the summary is 80% likely to get a score of 5, 10% likely to get a score of 4, 5% likely to get a score of 3, 3% likely to get a score of 2, and 2% likely to get a score of 1, you should rate the summary as 80, 10, 5, 3, 2.
    4. Only report back with your final rating. For example: 80, 10, 5, 3, 2

    Here is the article: {article}

    Here is the summary: {summary}
    """.format(article=article, summary=summary)

    return prompt


def rankPrompt_lvl0(article, summary1, summary2):
    prompt = """

    You will be given two summaries written for a news article. Your task is to rank the summaries based on the following criteria:

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summaries and compare them to the news article. Check if the summaries covers the main topic and key points of the news article, and if they resent them in a clear and logical order.
    3. Rank the summaries based on how likely you would recommend them to others. For example, if you think the second summary is more likely to be recommended than the first summary, you should rank them as: summary 2, summary 1.
    4. Only report back with your final ranking. For example: summary 2, summary 1

    Here is the article: {article}

    Here is the summary 1: {summary1}

    Here is the summary 2: {summary2}

""".format(article=article, summary1=summary1, summary2=summary2)
    return prompt


def rankPrompt_lvl1(article, summary1, summary2):
    prompt = """

    You will be given two summaries written for a news article. Your task is to rank the summaries based on the following criteria:
    
    Rank the summaries based on how they meet the criteria defined here. For example, as long as summary 2 better meets the criteira, rank: summary 2, summary 1.

    Output Format: 
    Only report back with your final ranking. For example: summary 2, summary 1

    Evaluation Criteria:
    1. Read the news article carefully and identify the main topic and key points.
    2. Read the summaries and compare them to the news article. Check if the summaries covers the main topic and key points of the news article, and if they resent them in a clear and logical order.
    3. If both summaries meet the requirements above, then rank them based on how concise and clear they are. For example, both summary 1 and summary 2 cover the key points of a news article, but summary 2 is shorter in lenght and summary 2 uses more common language, you should recommend summary 2 over summary 1.
    4. If both summaries meet the requirements above, then rank them based on whether they are free from grammatical errors and typos. For example, if the first summary has more grammatical errors and typos than summary 2, then you should recommend summary 2 over summary 1.
    5. If both summaries meet the requirements above, then rank them based on how likely you would recommend them to others. For example, if you think the summary 2 is more likely to be recommended than the first summary, you should rank them as: summary 2, summary 1.

    Here is the article: {article}

    Here is the summary 1: {summary1}

    Here is the summary 2: {summary2}

""".format(article=article, summary1=summary1, summary2=summary2)

    return prompt


def build(ckpt_dir, tokenizer_path, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=8000,
        max_batch_size=max_batch_size,
    )
    return generator


def call(generator, user_input):
    dialogs: List[Dialog] = []

    dialogs.append([{"role": "user", "content": "{}".format(user_input)}])
        
    results = generator.chat_completion(
        dialogs,
        max_gen_len = None,
        temperature=0,
    )

    final = ""
    for result in results:
        final=result['generation']['content']
    
    return final
        

def main(
    prompt_only: bool = False,
    prompt_type: str = "rateGood",
    summary1: str = "paraphrased",
    summary2: str = "original",
    ckpt_dir: str = "llama-2-7b-chat",
    tokenizer_path: str = "tokenizer.model",
    max_batch_size: int = 2,
):

    promptGenerator = {"rateGood": ratePromptGood, "rateBad": ratePromptBad, "rank_lvl0": rankPrompt_lvl0, "rank_lvl1": rankPrompt_lvl1}

    if prompt_type not in promptGenerator:
        print("Invalid prompt type!", prompt_type, "Valid prompt types are:")
        print(list(promptGenerator.keys()))
        sys.exit(1)

    if summary1 not in summaries or summary2 not in summaries:
        print("Contains invalid summaries! Valid summaries are:")
        print(list(summaries.keys()))
        sys.exit(1)        

    prompt = ''
    if prompt_type.startswith("rate"):
        prompt = promptGenerator[prompt_type](article, summaries[summary1])
    else:
        prompt = promptGenerator[prompt_type](article, summaries[summary1], summaries[summary2])

    if prompt_only:
        print(prompt)
        sys.exit(0)

    generator = build(ckpt_dir, tokenizer_path, max_batch_size)
    print(call(generator, prompt))


if __name__ == "__main__":
    fire.Fire(main)