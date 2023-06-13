import torch
# import gdown
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors

from tqdm import tqdm
from urllib.parse import urlencode
from annotated_text import annotated_text
from torch.utils.data import Dataset, DataLoader
from models.collate import collate_fn, gold_labels
from models.model import SpanAsteModel
from models.metrics import SpanEvaluator
from typing import Text, List, Any
from transformers import BertTokenizer, BertModel
from utils.processor import Res15DataProcessor, InputExample, DataProcessor
from utils.tager import RelationLabel, SpanLabel
from transformers import BertTokenizer

device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()


@st.cache
def load_model():
    # url = 'https://drive.google.com/uc?id=14qaP5Zh-ox0WYLRL3fBo3hjN7E8LRNwR'
    # output = 'model.pt'
    # gdown.download(url, output, quiet=False)

    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/VNRKw04VDWjwzg'  # Сюда вписываете вашу ссылку

    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Загружаем файл и сохраняем его
    download_response = requests.get(download_url)
    with open('model.pt', 'wb') as f:  # Здесь укажите нужный путь к файлу
        f.write(download_response.content)
load_model()

class CustomDatasetSample(Dataset):
    """
    An customer class representing txt data reading
    """

    def __init__(self,
                 sample: List,
                 processor: "DataProcessor",
                 tokenizer: "BertTokenizer",
                 max_seq_length: "int"
                 ) -> "None":
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sentence_list = []
        lines = self.get_lines(sample)
        self.examples = Res15DataProcessor(tokenizer, max_seq_length)._create_examples(lines, 'test')

    def __getitem__(self, idx: "int"):
        example = self.examples[idx]  # type:InputExample
        inputs = self.tokenizer(example.text_a, max_length=self.max_seq_length, padding='max_length', truncation=True)
        text = example.text_a
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        token_type_ids = inputs.token_type_ids
        spans = example.spans
        relations = example.relations
        span_labels = example.span_labels
        relation_labels = example.relation_labels
        seq_len = len([i for i in input_ids if i != 0])

        return text, input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len

    def __len__(self):
        return len(self.examples)

    @classmethod
    def get_lines(cls, samples: List) -> List:
        return [{"text": row.strip(), "labels": []} for row in samples]


def get_texts_from_inds(input_ids, relations_pred, gold_relation_indices, tonality: str) -> Any:
    if tonality == 'POS':
        val = 1
    elif tonality == 'NEG':
        val = 2
    else:
        val = 0
    relations_pred_inds = (relations_pred == val).nonzero(as_tuple=True)[-1]
    preds_tokens = torch.tensor(gold_relation_indices[0])[relations_pred_inds]
    preds_texts = []
    triplets_ids = []
    tokens_ids = []
    for pred_tokens in preds_tokens:
        aspect_inds = (int(pred_tokens[0]), int(pred_tokens[1]))
        opinion_inds = (int(pred_tokens[2]), int(pred_tokens[3]))
        aspect_tokens = input_ids[0][pred_tokens[0]:pred_tokens[1]]
        opinion_tokens = input_ids[0][pred_tokens[2]:pred_tokens[3]]
        triplets_ids.append(
            {
                'aspect': (aspect_inds[0], aspect_inds[1]),
                'opinion': (opinion_inds[0], opinion_inds[1])
            }
        )
        tokens_ids.extend(list(range(aspect_inds[0], aspect_inds[1])))
        tokens_ids.extend(list(range(opinion_inds[0], opinion_inds[1])))
        aspect = tokenizer.decode(aspect_tokens)
        opinion = tokenizer.decode(opinion_tokens)
        preds_texts.append((aspect, opinion, tonality))
    return triplets_ids, tokens_ids, preds_texts


def get_labels_texts(input_ids, relations_labels, relation_indices) -> Any:
    pos_triplets, pos_tokens_ids, pos_texts = get_texts_from_inds(input_ids, relations_labels, relation_indices, 'POS')
    neg_triplets, neg_tokens_ids, neg_texts = get_texts_from_inds(input_ids, relations_labels, relation_indices, 'NEG')
    triplets_ids = {'POS': pos_triplets, 'NEG': neg_triplets}
    tokens_ids = {'POS': pos_tokens_ids, 'NEG': neg_tokens_ids}
    return pos_texts + neg_texts, triplets_ids, tokens_ids

def format_color_groups(df, color):
    x = df.copy()
    i = 0
    for factor in color:
        x.iloc[i, :-1] = ''
        style = f'background-color: {color[i]}'
        x.loc[i, 'display color as background'] = style
        i = i + 1
    return x


colors = {
    'name': mcolors.CSS4_COLORS.keys(),
    'hex': mcolors.CSS4_COLORS.values()
}
df_colors = pd.DataFrame(colors)
df_colors['rgb'] = df_colors['hex'].apply(mcolors.hex2color)
df_colors['rgb'] = df_colors['rgb'].apply(lambda x: [round(c, 5) for c in x])
colors = df_colors['hex'].tolist()
random.shuffle(colors)


def set_color(triplets, colors=colors):
    for ton in triplets.keys():
        for i in range(len(triplets[ton])):
            triplets[ton][i]['color'] = colors[i]


def out_preprocess(input_ids, tokens, triplets, tokenizer) -> List:
    res = []
    aste_inds = tokens['POS'] + tokens['NEG']
    skip = []
    between = []
    for i in range(len(input_ids[0])):
        if input_ids[0][i] in [101, 102, 0]:
            continue
        if i in skip:
            continue
        if i in aste_inds:
            skip_flag = False
            res.append(tokenizer.decode(between))
            between.clear()
            for ton, trips in triplets.items():
                for triplet in trips:
                    for type_, inds in triplet.items():
                        if type_ == 'color' or skip_flag:
                            continue
                        start, end = inds
                        skip = list(range(start, end))
                        if i in skip:
                            word = tokenizer.decode(input_ids[0][start:end])
                            res.append((word, ton, triplet['color']))
                            skip_flag = True
        else:
            between.append(input_ids[0][i])
    if between:
        res.append(tokenizer.decode(between))
    return res


PARAMS = {
    'bert_model': 'ai-forever/ruBert-base',
    'batch_size': 1,
    'learning_rate': 5e-5,
    'weight_decay': 1e-2,
    'warmup_proportion': 0.1,
    'train_path': 'data/bank_3200',
    'dev_path': 'data/bank_3200',
    'save_dir': './checkpoint',
    'max_seq_len': 512,
    'num_epochs': 20,
    'seed': 0,
    'logging_steps': 50,
    'valid_steps': 50,
    'init_from_ckpt': None,
    'text_type': 'full'
}

MODEL_PATH = '/Users/maksimseleznev/Desktop/aste-app/checkpoint/sber-bert-base-st3200-bs1-lr5e-05'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(PARAMS['seed'])

# tokenizer
tokenizer = BertTokenizer.from_pretrained(PARAMS['bert_model'])

# create processor
processor = Res15DataProcessor(tokenizer, PARAMS['max_seq_len'])

# Init model
target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
model = SpanAsteModel(
        PARAMS['bert_model'],
        BertModel,
        target_dim,
        relation_dim,
        device=device
    )
model.load_state_dict(torch.load("model.pt", map_location=torch.device(device)))
model.to(device)

metric = SpanEvaluator()

model.eval()
metric.reset()

# input
sample = [st.text_input("Enter text")]

@st.cache
def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("model/model.pt")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)

    torch.load("model.pt", map_location=torch.device(device))
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model


if '.' not in sample[0][-2:]:
    sample[0] += '.'

lines = CustomDatasetSample.get_lines(sample)
sample_dataset = CustomDatasetSample(sample, processor, tokenizer, PARAMS['max_seq_len'])
sample_dataloader = DataLoader(sample_dataset, batch_size=PARAMS['batch_size'], collate_fn=collate_fn)
preds = []
with torch.no_grad():
    for batch_ix, batch in enumerate(tqdm(sample_dataloader)):
        curr_result = {
            'text': lines[batch_ix]['text'],
            'preds': []
        }
        text, input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)

        # forward
        spans_probability, span_indices, relations_probability, candidate_indices = model(
            input_ids, attention_mask, token_type_ids, seq_len)

        gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
        gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)

        num_correct, num_infer, num_label = metric.compute(relations_probability.cpu(),
                                                           torch.tensor(gold_relation_labels))
        relations_pred = relations_probability.cpu().argmax(-1)
        curr_result['preds'], triplets, tokens = get_labels_texts(
            input_ids, relations_pred, gold_relation_indices
        )
        set_color(triplets)
        out = out_preprocess(input_ids, tokens, triplets, tokenizer)
        preds.append(out)

annotated_text(*preds[0])

# annotated_text(
#     "This ",
#     ("is", "verb"),
#     " some ",
#     ("annotated", "adj"),
#     ("text", "noun"),
#     " for those of ",
#     ("you", "pronoun"),
#     " who ",
#     ("like", "verb"),
#     " this sort of ",
#     ("thing", "noun"),
#     "."
# )

