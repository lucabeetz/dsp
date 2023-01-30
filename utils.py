import re
import string
import unicodedata
import requests
import pandas as pd
from tqdm import tqdm
from IPython.display import display


def evaluate(run_chain, dev):
    data = []

    for example in tqdm(dev):
        prediction = run_chain(example)
        prediction = prediction.strip() 

        d = dict(example)
        d['prediction'] = prediction
        d['correct'] = EM(prediction, example['answers'])

        data.append(d)

    df = pd.DataFrame(data)
    correct = df['correct'].sum()
    print(f'Correct: {correct} / {len(df)}')

    pd.options.display.max_colwidth = None
    
    df['correct'] = df['correct'].apply(lambda x: '✅' if x else '❌')
    display(df)


def EM(prediction, answers_list):
    assert type(answers_list) == list

    return max(em_score(prediction, ans) for ans in answers_list)


def em_score(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=10):
        topk = colbertv2_get_request(self.url, query, k)

        topk = [doc['text'] for doc in topk]
        return topk


def colbertv2_get_request(url: str, query: str, k: int):
    payload = {'query': query, 'k': k}
    res = requests.get(url, params=payload)

    topk = res.json()['topk'][:k]
    return topk
