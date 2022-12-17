import numpy as np
import random
import scipy.stats
import math
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm
import pickle
from wikidataintegrator import wdi_core
from wikidata.client import Client
import wikidata
from itertools import compress

import en_core_web_sm
import spacy
import warnings
warnings.filterwarnings("ignore")

# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

#nlp = en_core_web_sm.load()
client = wikidata.client.Client()
softmax = nn.Softmax()

def get_description_name(idd):
    '''
    This function returns a name of an entity and its description given WikiData id
    
        input:  (str) wikidata id, e.x. 'Q2'
        output: (str) concatenated 'name, description' of a given entity
    '''
    entity = client.get(idd, load=True)
    name = "None"
    description = "None"
    try:
        name = entity.data["labels"]["en"]["value"]
        
    except:
        pass
    return name



    
    
    
### For top k sample
    
def getScores(ids, scores, pad_token_id):
    """get sequence scores from model.generate output"""
    scores = torch.stack(scores, dim=1)
    log_probs = torch.log_softmax(scores, dim=2)
    # remove start token
    ids = ids[:,1:]
    # gather needed probs
    x = ids.unsqueeze(-1).expand(log_probs.shape)
    needed_logits = torch.gather(log_probs, 2, x)
    final_logits = needed_logits[:, :, 0]
    padded_mask = (ids == pad_token_id)
    final_logits[padded_mask] = 0
    final_scores = final_logits.sum(dim=-1)
    return final_scores.cpu().detach().numpy()

def topkSample(input, model, tokenizer, 
                num_samples=5,
                num_beams=1,
                max_output_length=128):
    tokenized = tokenizer(input, return_tensors="pt")
    tokenized.to(device)
    out = model.generate(**tokenized,
                        do_sample=True,
                        num_return_sequences = num_samples,
                        num_beams = num_beams,
                        eos_token_id = tokenizer.eos_token_id,
                        pad_token_id = tokenizer.pad_token_id,
                        output_scores = True,
                        return_dict_in_generate=True,
                        max_length=max_output_length,)
    out_tokens = out.sequences
    out_str = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
    out_scores = getScores(out_tokens, out.scores, tokenizer.pad_token_id)
    
    pair_list = [(x[0], x[1]) for x in zip(out_str, out_scores)]
    sorted_pair_list = sorted(pair_list, key=lambda x:x[1], reverse=True)
    return sorted_pair_list

def greedyPredict(input, model, tokenizer):
    input_ids = tokenizer([input], return_tensors="pt").input_ids
    out_tokens = model.generate(input_ids)
    out_str = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
    return out_str[0]


def check_t5_sq(t5_tok, t5_qa_model):
    
    sq_test_data = np.load("simple_questions_test.npy")
    questions = sq_test_data[:,3]


    answers = []
    for i in tqdm(range(len(questions))):
        input_ids = t5_tok(questions[i], return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        gen_output = t5_qa_model.generate(input_ids)[0]
        answers.append(gen_output)


    preds_sq = []
    for x in answers:
        preds_sq.append(t5_tok.decode(x, skip_special_tokens=True))


    preds_id_sq = []
    for i in range(len(preds_sq)):
        try:
            x = from_text_to_id(preds_sq[i])
        except:
            x = "None"

        preds_id_sq.append(x)

    right_sq = 0
    for i in tqdm(range(len(preds_id_sq))):
        if preds_id_sq[i] == sq_test_data[i,2]:
            right_sq += 1
        else:
            pass

    acc = right_sq/len(preds_id_sq)
    
    return acc, preds_id_sq


def first_letter_big(word):
    try:
        return word[0].upper() + word[1:]
    except:
        return word

def stable_experiments(seed_num):
    
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    
    
