import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# function to install packages from .py file
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

init_dir = os.getcwd()
# import libraries
import gdown
import json
import pickle
import torch
import gc
import re
install("wikidata")
from wikidata.client import Client
install("pandas")
install("scipy")
install("matplotlib")

import pandas as pd
import numpy as np
import random
import scipy.stats
import string
import json
from itertools import compress
from datetime import datetime
import matplotlib.pyplot as plt

install("stanza")
import stanza
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("All libraries are imported")

print("Upload data and model")
# model
os.system("rm -rf fairseq_multilingual_entity_disambiguation*")
os.system("wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz")
os.system("tar -xvf fairseq_multilingual_entity_disambiguation.tar.gz")
# data
os.system("wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl")
os.system("wget http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl")


os.chdir(init_dir)

#Quick Start. Multilingual Autoregressive Entity Linking. GENRE

path_1 = init_dir+'/GENRE'
path_faiseq = init_dir + "/fairseq"
#sys.path.insert(0, path_1)
from genre.trie import Trie
from genre.trie import MarisaTrie

print("Trie, MarisaTrie and mGENRE are imported")

#sys.path.insert(0, init_dir)
with open("lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

# with open("titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

with open("titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

print("lang_title2wikidataID and trie done")
sys.path.insert(0, path_faiseq)
from genre.fairseq_model import mGENRE

device =  "cpu"#torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# generate Wikipedia titles and language IDs
model_mGENRE = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation").eval()
model_mGENRE.to(device)


sentences = ["[START] The founder of the theory of relativity [END] received the Nobel Prize."]

print("first experiment")
print(sentences)
print(model_mGENRE.sample(
    sentences,
    beam = 5,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist())
        if e < len(model_mGENRE.task.target_dictionary)
    ],
    text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
    marginalize=True,
    verbose = True
))

#### Uncertinty estimation
### Data uploading (Simple Questions)
os.chdir(init_dir)
## We are using test dataset
path_to_train_simple_questions = init_dir+"/annotated_wd_data_test_answerable.txt"
data = pd.read_table(path_to_train_simple_questions, header=None).rename(columns = {0:"subject", 1:"property", 2:"object", 3:"question"})
## Data uploading (Simple Questions with 1 answer)
path_to_annotated_wd_data_test_answerable_answer_counts = init_dir+"/annotated_wd_data_test_answerable_answer_counts.txt"
data_clean = pd.read_table(path_to_annotated_wd_data_test_answerable_answer_counts, header=None).rename(columns = {0:"subject", 1:"property", 2:"object", 3:"question"})
data_clean.rename(columns = {4: "number_of_answers"}, inplace = True)
data_clean = data_clean[data_clean["number_of_answers"] == 1]
data_clean = data_clean.drop(["number_of_answers"], axis = 1)

## MC Dropout
print("#"*100)
print("MC Dropout")
print("#"*100)

device =  "cpu"#torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("device is ", device)

print(os.getcwd())
os.chdir(init_dir)

from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE

model_mGENRE_mcdropout = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation",
                                                dropout = 0.1, attention_dropout = 0.1)
model_mGENRE_mcdropout.to(device)
### switch to train mode
model_mGENRE_mcdropout.train()

sentences = ["[START] Пушкин [END] был поэтом и писателем."]

print(model_mGENRE_mcdropout.models[0].encoder.dropout_module.training, "\n")

for i in range(3):
    print(model_mGENRE_mcdropout.sample(
        sentences,
        beam=3,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist())
            if e < len(model_mGENRE_mcdropout.task.target_dictionary)
        ],
        text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
        marginalize=True,
        seed=i
    ), "\n")

print(model_mGENRE_mcdropout.models[0].encoder.dropout_module.training, "\n")


### Number of functions to count uncertainty
def BALD_count(
        data,
        model,
        number_of_models_in_ensemble=3,
        beams=5,
        marginalize=True,
        only_BALD=True
):
    '''
    Bayesian Active Learning by Disagreement (BALD)

    '''

    questions_to_model = data
    number_of_questions = len(questions_to_model)
    all_results = []

    print("Generating samples on received text")
    for i in range(number_of_models_in_ensemble):
        random.seed(i)
        torch.manual_seed(i)
        np.random.seed(i)
        all_results.append(model.sample(questions_to_model,
                                        beam=beams,
                                        prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                            e for e in trie.get(sent.tolist())
                                            if e < len(model.task.target_dictionary)
                                        ],
                                        text_to_id=lambda x: max(
                                            lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                                            key=lambda y: int(y[1:])),
                                        marginalize=marginalize,
                                        verbose=True,
                                        seed=i))

    question_sorted = [np.array(all_results)[:, i] for i in range(number_of_questions)]

    set_of_ids = []

    for quest in range(number_of_questions):
        flat = [x for xs in list(question_sorted[quest]) for x in xs]
        set_of_ids.append(set().union([i["id"] for i in flat]))

    probs_and_ids_matrix = [[[{d["id"]: np.exp((float(d["score"]) * (len(d["texts"][0][:-6].split(" "))) ** 0.5))}
                              for d in i[j]]
                             for j in range(len(i))]
                            for i in all_results]

    matrix_of_probs_and_ids = []
    for seed in probs_and_ids_matrix:
        line_of_probs_and_ids = []
        for sentence in seed:
            line_of_probs_and_ids.append({k: v for d in sentence for k, v in d.items()})

        matrix_of_probs_and_ids.append(line_of_probs_and_ids)

    final_matrix_of_probs = matrix_of_probs_and_ids.copy()
    for seed in matrix_of_probs_and_ids:
        for n, sentence in enumerate(seed):
            for id_ in set_of_ids[n]:
                if id_ not in list(sentence.keys()):
                    sentence[id_] = 0.0

    final_probs = []

    for i_quest in range(number_of_questions):
        x = {}
        for y in np.array(matrix_of_probs_and_ids)[:, i_quest]:
            x = {k: x.get(k, 0) + y.get(k, 0) / number_of_models_in_ensemble for k in set(x) | set(y)}
        final_probs.append(x)

    print("final_probs = ", final_probs)
    first_sum = [scipy.stats.entropy(list(x.values())) for x in final_probs]

    final_probs_2_sum = []

    for i_quest in range(number_of_models_in_ensemble):
        # seeds
        new = []
        for quest in range(number_of_questions):
            # questions
            new.append(scipy.stats.entropy(list(final_matrix_of_probs[i_quest][quest].values())))
        final_probs_2_sum.append(new)

    second_sum = np.array(final_probs_2_sum).mean(axis=0)

    # final part
    predicted_entropy = np.array(first_sum)
    expected_entropy = np.array(second_sum)
    BALD = predicted_entropy - expected_entropy

    if only_BALD:
        return BALD
    else:
        return BALD, predicted_entropy, expected_entropy

### Uncertainty estimation experiments
## NER for text preparation
stanza.download('en')

# description of available
print(os.system("nvidia-cdl"))

device = "cpu"#torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model_mGENRE_mcdropout = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation",
                                                dropout = 0.1, attention_dropout = 0.1)

model_mGENRE_mcdropout.to(device)
# switch to train mode
model_mGENRE_mcdropout.train()

### Uncertainty functions

def stanza_nlp(text, device, language):
    nlp = stanza.Pipeline(lang=language, processors='tokenize,ner', verbose=False, use_gpu=False)
    doc = nlp(text)
    return [ent.text for sent in doc.sentences for ent in sent.ents]


def NER_Stanza(sentence, language, device=device):
    res = stanza_nlp(text=sentence, device=device, language=language)
    if res != []:
        if len(res) == 1:
            first_part, second_part = sentence.split(res[0])[0], sentence.split(res[0])[1]
            output = first_part + "[START] " + res[0] + " [END]" + second_part
            return output
        else:
            for i in range(len(res)):
                output = ' '.join(['[START] {} [END]'.format(x) if x in res else x for x in sentence.split(" ")])
            return output

    else:
        return sentence


def UE_estimate(
        data,
        model,
        ue_metrics=['entropy', 'maxprob', 'delta', 'BALD', 'expected entropy', 'predicted entropy'],
        number_of_samples=100,
        beams=5,
        seed=13,
        task="Question Answering (object detection)",
        target_col="object",
        NER=None,
        dataset="Simple Questions",
        language="en"
):
    n = number_of_samples
    rang = range(n)
    df = data.sample(n=n, replace=False, random_state=seed)

    # Simple Questions data set (only enlish column)
    if dataset == "Simple Questions":
        pass

    # RUBQ 2.0 (enlich and rusian columns)
    elif dataset == "RUBQ 2.0":
        if language == "ru":
            df["question"] = df["question_ru"]
            df.drop(["question_ru", "question_en"], axis=1, inplace=True)

        if language == "en":
            df["question"] = df["question_en"]
            df.drop(["question_ru", "question_en"], axis=1, inplace=True)

    elif dataset == "Mewsli9":
        pass

    if NER == None:
        df = df.reset_index().drop(['index'], axis=1)



    elif NER == "Stanza":
        df = df.reset_index().drop(['index'], axis=1)
        da = pd.DataFrame(df['question'].apply(lambda x: string.capwords(x)))
        print("Started preparing text using NER")
        for i in tqdm(range(len(da))):
            # print("before: ", da.loc[i, "question"])
            # print("before df: ", df.loc[i, "question"])
            da.loc[i, "question"] = NER_Stanza(da.loc[i, "question"], language)

            # print("after: ", da.loc[i, "question"])
            # print("correct answer: ", df.loc[i, "subject"])

        df["question"] = da["question"]
        print("Finished preparing text using NER")

    print("Started sampling variants using mGENRE")
    model_mGENRE_mcdropout_result = model.sample(list(df['question']),
                                                 beam=beams,
                                                 prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                                     e for e in trie.get(sent.tolist())
                                                     if e < len(model.task.target_dictionary)
                                                 ],
                                                 text_to_id=lambda x: max(
                                                     lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                                                     key=lambda y: int(y[1:])),
                                                 marginalize=True,
                                                 verbose=True,
                                                 seed=seed)

    all_results = [model_mGENRE_mcdropout_result]

    probs_and_ids_matrix = [[[{d["id"]: np.exp((float(d["score"]) * (len(d["texts"][0][:-6].split(" "))) ** 0.5))}
                              for d in i[j]]
                             for j in range(len(i))]
                            for i in all_results]

    probs_for_examples = [[float(list(i[j].values())[0]) for j in range(len(i))] for i in probs_and_ids_matrix[0]]

    quants = [thresh / 20 for thresh in range(1, 21)]
    quants_more = [thresh / 20 for thresh in range(0, 20)]

    predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]
    y_pred = predictions
    y_true = list(df[target_col])

    result = [x in y_pred for x in y_true]
    accuracy_on_full_data = np.round(sum(result) / len(result), 4) * 100

    if 'entropy' in ue_metrics:
        print("entropy")

        entropies = [scipy.stats.entropy(i) for i in probs_for_examples]
        thresholds_entropy = [np.quantile(entropies, q) for q in quants]

        predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]
        accuracy_entropy = []
        share_of_observations_entropy = []

        for threshold in thresholds_entropy:
            list_a = list(rang)
            fil = entropies <= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_entropy.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_entropy.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

    if 'maxprob' in ue_metrics:
        print("maxprob")

        probas = [abs(i[0]) for i in np.array([np.exp(np.array(i)) for i in probs_for_examples])]
        thresholds_probas = [np.quantile(probas, q) for q in quants_more]

        predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]
        accuracy_probas = []
        share_of_observations_probas = []

        for threshold in thresholds_probas:
            list_a = list(rang)
            fil = probas >= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_probas.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_probas.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

    if 'delta' in ue_metrics:

        print("delta")
        deltas = [abs(i[0] - i[1]) if len(i) > 1 else abs(i[0]) for i in
                  np.array([np.exp(np.array(i)) for i in probs_for_examples])]

        thresholds_deltas = [np.quantile(deltas, q) for q in quants_more]

        predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]
        accuracy_delta = []
        share_of_observations_delta = []

        for threshold in thresholds_deltas:
            list_a = list(rang)
            fil = deltas >= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_delta.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_delta.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

    if ('BALD' in ue_metrics) & ('expected entropy' in ue_metrics) & ('predicted entropy' in ue_metrics):

        start_time = datetime.now()

        BALD, predicted_entropy, expected_entropy = BALD_count(
            data=list(df['question']),
            model=model_mGENRE_mcdropout,
            number_of_models_in_ensemble=5,
            beams=5,
            only_BALD=False
        )

        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        BALD_thresholds = [np.quantile(BALD, q) for q in quants]
        predicted_entropy_thresholds = [np.quantile(predicted_entropy, q) for q in quants]
        expected_entropy_thresholds = [np.quantile(expected_entropy, q) for q in quants]

        predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]

        accuracy_BALD = []
        accuracy_predicted_entropy = []
        accuracy_expected_entropy = []

        share_of_observations_BALD = []
        share_of_observations_predicted_entropy = []
        share_of_observations_expected_entropy = []

        print("BALD")
        for threshold in BALD_thresholds:
            list_a = list(rang)
            fil = BALD <= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_BALD.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_BALD.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

        print("\n")
        print("predicted_entropy")
        for threshold in predicted_entropy_thresholds:
            list_a = list(rang)
            fil = predicted_entropy <= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_predicted_entropy.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_predicted_entropy.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

        print("\n")
        print("expected_entropy")
        for threshold in expected_entropy_thresholds:
            list_a = list(rang)
            fil = expected_entropy <= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_expected_entropy.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_expected_entropy.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

    if ('BALD' in ue_metrics) & ('expected entropy' not in ue_metrics) & ('predicted entropy' not in ue_metrics):

        start_time = datetime.now()

        BALD, predicted_entropy, expected_entropy = BALD_count(
            data=list(df['question']),
            model=model_mGENRE_mcdropout,
            number_of_models_in_ensemble=5,
            beams=5,
            only_BALD=False
        )

        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        BALD_thresholds = [np.quantile(BALD, q) for q in quants]
        predicted_entropy_thresholds = [np.quantile(predicted_entropy, q) for q in quants]
        expected_entropy_thresholds = [np.quantile(expected_entropy, q) for q in quants]

        predictions = [i[0]['id'] for i in model_mGENRE_mcdropout_result]

        accuracy_BALD = []
        accuracy_predicted_entropy = []
        accuracy_expected_entropy = []

        share_of_observations_BALD = []
        share_of_observations_predicted_entropy = []
        share_of_observations_expected_entropy = []

        print("BALD")
        for threshold in BALD_thresholds[:]:
            list_a = list(rang)
            fil = BALD <= threshold

            y_pred = list(compress(predictions, fil))
            y_true = list(df.iloc[list(compress(list_a, fil)), :][target_col])

            result = [x in y_pred for x in y_true]
            accuracy = np.round(sum(result) / len(result), 4) * 100
            accuracy_BALD.append(accuracy)
            share = np.round(len(result) / n * 100, 4)
            share_of_observations_BALD.append(share)

            print("threshold = ", format(threshold, '.2f'), "\t",
                  "accuracy = ", format(accuracy, '.2f'), "%\t",
                  "number of observations = ", len(result), '\t',
                  "share of observations = ", format(share, '.2f'), "%")

        print("\n")

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16
            }

    font_title = {'family': 'serif',
                  'color': 'darkred',
                  'weight': 'normal',
                  'size': 20
                  }

    plt.figure(figsize=(16, 12))
    plt.xlabel("Rejection rate, %", fontdict=font)
    plt.ylabel("Accuracy, %", fontdict=font)
    plt.xticks(ticks=np.arange(0, 100, step=5))
    plt.grid(color='black', linewidth=0.15)

    if NER is None:
        plt.title(
            "Uncertainty Estimation for mGENRE based on {} {} {} examples. \n {} w/o NER".format(number_of_samples,
                                                                                                 dataset, language,
                                                                                                 task),
            fontdict=font_title)
    else:
        plt.title("Uncertainty Estimation for mGENRE based on {} {} {} examples. \n {} with NER - {}".format(
            number_of_samples, dataset, language, task, NER), fontdict=font_title)

    outs = {}

    if 'entropy' in ue_metrics:
        outs["entropy"] = (list(100.0 - np.array(share_of_observations_entropy)), accuracy_entropy)
        plt.plot(100.0 - np.array(share_of_observations_entropy), accuracy_entropy, label="Entropy (single model)",
                 c="red", marker='.', markersize=13, linewidth=0.8);

    if 'expected entropy' in ue_metrics:
        plt.plot(100.0 - np.array(share_of_observations_expected_entropy), accuracy_expected_entropy,
                 label="Expected Entropy", c="blue", marker='.', markersize=13, linewidth=0.8);
        outs["expected entropy"] = (
        list(100.0 - np.array(share_of_observations_expected_entropy)), accuracy_expected_entropy)

    if 'predicted entropy' in ue_metrics:
        plt.plot(100.0 - np.array(share_of_observations_predicted_entropy), accuracy_predicted_entropy,
                 label="Predicted Entropy", c="green", marker='.', markersize=13, linewidth=0.8);
        outs["predicted entropy"] = (
        list(100.0 - np.array(share_of_observations_predicted_entropy)), accuracy_predicted_entropy)

    if 'BALD' in ue_metrics:
        plt.plot(100.0 - np.array(share_of_observations_BALD), accuracy_BALD, label="BALD", c="black", marker='.',
                 markersize=13, linewidth=0.8);
        outs["BALD"] = (list(100.0 - np.array(share_of_observations_BALD)), accuracy_BALD)

    if 'delta' in ue_metrics:
        plt.plot(100.0 - np.array(share_of_observations_delta), accuracy_delta, label="Delta", c="pink", marker='.',
                 markersize=13, linewidth=0.8);
        outs["delta"] = (list(100.0 - np.array(share_of_observations_delta)), accuracy_delta)

    if 'maxprob' in ue_metrics:
        plt.plot(100.0 - np.array(share_of_observations_probas), accuracy_probas, label="Maxprob", c="orange",
                 marker='.', markersize=13, linewidth=0.8);
        outs["maxprob"] = (list(100.0 - np.array(share_of_observations_probas)), accuracy_probas)

    plt.annotate("{}".format(np.round(accuracy_on_full_data, 2)), (0 - 3.7, accuracy_on_full_data));

    plt.legend(fontsize=16);

    if NER is None:
        plt.savefig('UE_mGENRE_{}_obserations_{}_{}_{}_wo_NER.png'.format(number_of_samples, dataset, language, task), dpi=150);
    else:
        plt.savefig('UE_mGENRE_{}_obserations_{}_{}_{}_with_NER.png'.format(number_of_samples, dataset, language, task), dpi=150);

    return outs

# Press the green button in the gutter to run the script.

### 5. (SUBJECT) EL with mGENRE with NER Stanza on Simple Questions where only 1 answer
print("5. (SUBJECT) EL with mGENRE with NER Stanza on Simple Questions where only 1 answer")

start_time_ue = datetime.now()

T_outs_3000_EL_with_NER_clean = UE_estimate(data=data_clean,
                                          model=model_mGENRE_mcdropout,
                                          number_of_samples=50,
                                          task='Entity Linking (subject detection)',
                                          target_col="subject",
                                          NER="Stanza",
                                          seed = 13)

end_time_ue = datetime.now()
print('Duration of uncertainty estimation: {}'.format(end_time_ue - start_time_ue))

with open('T_outs_3000_EL_with_NER_clean.json', 'w') as fp:
    json.dump(T_outs_3000_EL_with_NER_clean, fp)

print("Experiment successfully ended")

