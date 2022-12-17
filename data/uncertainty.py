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
softmax = nn.Softmax()

def first_letter_big(word):
    try:
        return word[0].upper() + word[1:]
    except:
        return word



def sinlge_model_ue(
    name_of_metric,
    name_of_dataset,
    probas,
    predictions,
    num_quantiles,
    answers,
    top_ks,
    sample_questions,
    plot_rejection_curves = True,
):

    quants = [thresh / num_quantiles for thresh in range(1, num_quantiles+1)]
    
    if name_of_metric == "entropy":
        ue_metrics = [scipy.stats.entropy(i) for i in probas]
    elif name_of_metric == "delta":
        ue_metrics = [i[0] - i[1] for i in probas]
        ue_metrics = torch.stack(ue_metrics).tolist()
    elif name_of_metric == "max_prob":
        ue_metrics = [i[0] for i in probas]
        ue_metrics = torch.stack(ue_metrics).tolist()
        
    
    thresholds_ue_metric = [np.quantile(ue_metrics, q) for q in quants]
    
    if (name_of_metric == "delta") or (name_of_metric == "max_prob"):
        thresholds_ue_metric = thresholds_ue_metric[::-1]

    accuracy_for_each_topk_ue_metrics = []

    #top_k pred
    for top_k in top_ks:

        accuracy_for_each_quantile = {}

        for quantile in range(num_quantiles):

            
            if name_of_metric == "entropy":
                # list of predictions
                list_of_predictions = list(compress(predictions, ue_metrics <= thresholds_ue_metric[quantile]))

                #list of correct answers
                list_of_correct_answers = list(compress(answers[:sample_questions], ue_metrics <= thresholds_ue_metric[quantile]))
                
            elif name_of_metric == "delta":
                
                # list of predictions
                list_of_predictions = list(compress(predictions, ue_metrics >= thresholds_ue_metric[quantile]))
                #print("len(list_of_predictions)", len(list_of_predictions))

                #list of correct answers
                list_of_correct_answers = list(compress(answers[:sample_questions], ue_metrics >= thresholds_ue_metric[quantile]))
                
            elif name_of_metric == "max_prob":
                
                
                list_of_predictions = list(compress(predictions, ue_metrics >= thresholds_ue_metric[quantile]))
                #print("len(list_of_predictions)", len(list_of_predictions))

                #list of correct answers
                list_of_correct_answers = list(compress(answers[:sample_questions], ue_metrics >= thresholds_ue_metric[quantile]))
        
                

            # choose top k predictions
            top_k_list_of_predictions = [i[:top_k] for i in list_of_predictions]
            #print("top_k_list_of_predictions", top_k_list_of_predictions, "\n")


            top_k_list_of_predicted_ids= []
            for sample in top_k_list_of_predictions:
                new = []
                for prediction in sample:
                    try:
                        x = from_text_to_id(prediction)
                    except:
                        x = "None"
                    new.append(x)

                top_k_list_of_predicted_ids.append(new)

            #print("top_k_list_of_predicted_ids", top_k_list_of_predicted_ids)

            right = 0
            for i in range(len(list_of_correct_answers)):
                if any(item in top_k_list_of_predicted_ids[i] for item in list_of_correct_answers[i]):
                    right += 1
                else:
                    pass

            accuracy = np.round(right/len(list_of_correct_answers), 4)*100
            #print(f"Top-{top_k} accuracy on the {(quantile+1)*(int(100/num_quantiles))}% the most confident from the entropy point of view = {accuracy}%")

            accuracy_for_each_quantile[(quantile+1)*(int(100/num_quantiles))] = accuracy
        accuracy_for_each_topk_ue_metrics.append(accuracy_for_each_quantile)
        
    
    if plot_rejection_curves:
        
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


        plt.figure(figsize = (15, 9))
        plt.title(f"Rejection curves based on {name_of_metric} for {name_of_dataset}", fontdict=font_title)

        plt.xlabel("Rejection rate, %", fontdict=font)
        plt.ylabel("Accuracy, %", fontdict=font)
        plt.xticks(ticks=np.arange(0, 100, step=5))
        plt.grid(color='black', linewidth=0.15)
        
        if 1 in top_ks:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_ue_metrics[0].keys())),
                np.array(list(accuracy_for_each_topk_ue_metrics[0].values())),
                label=f"Top-{1} Accuracy via {name_of_metric}", c="blue", marker='.', markersize=13, linewidth=0.8);
        
        if 2 in top_ks:
            
            plt.plot(100 - np.array(list(accuracy_for_each_topk_ue_metrics[1].keys())),
                np.array(list(accuracy_for_each_topk_ue_metrics[1].values())),
                label=f"Top-{2} Accuracy via {name_of_metric}", c="red", marker='.', markersize=13, linewidth=0.8);
        
        if 5 in top_ks:
            
            plt.plot(100 - np.array(list(accuracy_for_each_topk_ue_metrics[2].keys())),
                np.array(list(accuracy_for_each_topk_ue_metrics[2].values())),
                label=f"Top-{5} Accuracy via {name_of_metric}", c="green", marker='.', markersize=13, linewidth=0.8);
           
        if 10 in top_ks:
            
            plt.plot(100 - np.array(list(accuracy_for_each_topk_ue_metrics[3].keys())),
                np.array(list(accuracy_for_each_topk_ue_metrics[3].values())),
                label=f"Top-{10} Accuracy via {name_of_metric}", c="orange", marker='.', markersize=13, linewidth=0.8);

        if 20 in top_ks:
        
            plt.plot(100 - np.array(list(accuracy_for_each_topk_ue_metrics[4].keys())),
                np.array(list(accuracy_for_each_topk_ue_metrics[4].values())),
                label=f"Top-{20} Accuracy via {name_of_metric}", c="pink", marker='.', markersize=13, linewidth=0.8);

        if 1 in top_ks:
            accuracy_on_full_data_top_1 = np.array(list(accuracy_for_each_topk_ue_metrics[0].values()))[-1]
            plt.annotate("{}".format(np.round(accuracy_on_full_data_top_1, 2)), (0 - 4.5, accuracy_on_full_data_top_1));
        if 2 in top_ks:
            accuracy_on_full_data_top_2 = np.array(list(accuracy_for_each_topk_ue_metrics[1].values()))[-1]
            plt.annotate("{}".format(np.round(accuracy_on_full_data_top_2, 2)), (0 - 4.5, accuracy_on_full_data_top_2));
        if 5 in top_ks:
            accuracy_on_full_data_top_5 = np.array(list(accuracy_for_each_topk_ue_metrics[2].values()))[-1]
            plt.annotate("{}".format(np.round(accuracy_on_full_data_top_5, 2)), (0 - 4.5, accuracy_on_full_data_top_5));
        if 10 in top_ks:
            accuracy_on_full_data_top_10 = np.array(list(accuracy_for_each_topk_ue_metrics[3].values()))[-1]
            plt.annotate("{}".format(np.round(accuracy_on_full_data_top_10, 2)), (0 - 4.5, accuracy_on_full_data_top_10));
        if 20 in top_ks:
            accuracy_on_full_data_top_20 = np.array(list(accuracy_for_each_topk_ue_metrics[4].values()))[-1]
            plt.annotate("{}".format(np.round(accuracy_on_full_data_top_20, 2)), (0 - 4.5, accuracy_on_full_data_top_20));

        plt.legend(fontsize=16);

        plt.savefig(f"QA_Rejection_curves_based_on_{name_of_metric}_for_{name_of_dataset}_T5_XL_SSM_NQ_model.png", dpi = 200)

    
    return accuracy_for_each_topk_ue_metrics, ue_metrics


def predict_and_proba(question, model, tokenizer, device, num_beams, max_sequence_length = 32):
    """This function generates sequence = answer on given question

    1) it encodes qustion using tokenizer
    2) converts tokenized text to device
    3) generate answer with dredefined beam size using tokenized text
    4) as output we receive answers and their probabilities

    """

    input_ids = tokenizer([question], return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    out = model.generate(input_ids,
                         num_return_sequences = num_beams,
                         num_beams = num_beams,
                         eos_token_id = tokenizer.eos_token_id,
                         pad_token_id = tokenizer.pad_token_id,
                         output_scores = True,
                         return_dict_in_generate=True,
                         early_stopping=True,
                         max_length=max_sequence_length)

    prediction = [tokenizer.decode(out.sequences[i], skip_special_tokens=True) for i in range(num_beams)]
    prediction = [first_letter_big(word) for word in prediction]
    probs = softmax(out.sequences_scores)

    return prediction, probs


def aggregation_plot_sinle_measures(
    ue_metrics_to_plot: "list",
    top_k_to_plot: "list",
    name_of_dataset,
    accuracy_for_each_topk_delta = None,
    accuracy_for_each_topk_maxprob = None,
    accuracy_for_each_topk_entropy = None,
    model_name = "T5_XL_SSM_NQ"
):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 20
            }

    font_title = {'family': 'serif',
                  'color': 'darkred',
                  'weight': 'normal',
                  'size': 24
                  }

    plt.figure(figsize = (24, 20))
    plt.title(f"Rejection curves based on Single model UE measures for {name_of_dataset}", fontdict=font_title)
    plt.xlabel("Rejection rate, %", fontdict=font)
    plt.ylabel("Accuracy, %", fontdict=font)
    plt.xticks(ticks=np.arange(0, 100, step=5), size = 14)
    plt.yticks(size = 14)
    plt.grid(color='black', linewidth=0.15)

    linewidth = 1.8
    markersize = 9

    if 1 in top_k_to_plot:

        if "delta" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_delta[0].keys())),
                     np.array(list(accuracy_for_each_topk_delta[0].values())),
                     label=f"Top-{1} Accuracy via Delta", c="blue", marker='d', markersize=markersize, linewidth=1.5);

        if "max_prob" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_maxprob[0].keys())),
                     np.array(list(accuracy_for_each_topk_maxprob[0].values())),
                     label=f"Top-{1} Accuracy via Maxprob", c="blue", marker='.', markersize=markersize, linewidth=linewidth, linestyle = '--');

        if "entropy" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_entropy[0].keys())),
                 np.array(list(accuracy_for_each_topk_entropy[0].values())),
                 label=f"Top-{1} Accuracy via Entropy", c="blue", marker='*', markersize=markersize, linewidth=linewidth, linestyle = ':');

    if 2 in top_k_to_plot:

        if "delta" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_delta[1].keys())),
                 np.array(list(accuracy_for_each_topk_delta[1].values())),
                 label=f"Top-{2} Accuracy via Delta", c="red", marker='d', markersize=markersize, linewidth=linewidth);

        if "max_prob" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_maxprob[1].keys())),
                 np.array(list(accuracy_for_each_topk_maxprob[1].values())),
                 label=f"Top-{2} Accuracy via Maxprob", c="red", marker='.', markersize=markersize, linewidth=linewidth, linestyle = '--');

        if "entropy" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_entropy[1].keys())),
                 np.array(list(accuracy_for_each_topk_entropy[1].values())),
                 label=f"Top-{2} Accuracy via Entropy", c="red", marker='*', markersize=markersize, linewidth=linewidth, linestyle = ':');


    if 5 in top_k_to_plot:

        if "delta" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_delta[2].keys())),
                 np.array(list(accuracy_for_each_topk_delta[2].values())),
                 label=f"Top-{5} Accuracy via Delta", c="green", marker='d', markersize=markersize, linewidth=linewidth);

        if "max_prob" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_maxprob[2].keys())),
                 np.array(list(accuracy_for_each_topk_maxprob[2].values())),
                 label=f"Top-{5} Accuracy via Maxprob", c="green", marker='.', markersize=markersize, linewidth=linewidth, linestyle = '--');

        if "entropy" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_entropy[2].keys())),
                 np.array(list(accuracy_for_each_topk_entropy[2].values())),
                 label=f"Top-{5} Accuracy via Entropy", c="green", marker='*', markersize=markersize, linewidth=linewidth, linestyle = ':');


    if 10 in top_k_to_plot:

        if "delta" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_delta[3].keys())),
                 np.array(list(accuracy_for_each_topk_delta[3].values())),
                 label=f"Top-{10} Accuracy via Delta", c="orange", marker='d', markersize=markersize, linewidth=linewidth);

        if "max_prob" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_maxprob[3].keys())),
                 np.array(list(accuracy_for_each_topk_maxprob[3].values())),
                 label=f"Top-{10} Accuracy via Maxprob", c="orange", marker='.', markersize=markersize, linewidth=linewidth, linestyle = '--');

        if "entropy" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_entropy[3].keys())),
                 np.array(list(accuracy_for_each_topk_entropy[3].values())),
                 label=f"Top-{10} Accuracy via Entropy", c="orange", marker='*', markersize=markersize, linewidth=linewidth, linestyle = ':');

    if 20 in top_k_to_plot:

        if "delta" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_delta[4].keys())),
                 np.array(list(accuracy_for_each_topk_delta[4].values())),
                 label=f"Top-{20} Accuracy via Delta", c="pink", marker='d', markersize=markersize, linewidth=linewidth);

        if "max_prob" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_maxprob[4].keys())),
                 np.array(list(accuracy_for_each_topk_maxprob[4].values())),
                 label=f"Top-{20} Accuracy via Maxprob", c="pink", marker='.', markersize=markersize, linewidth=linewidth, linestyle = '--');

        if "entropy" in ue_metrics_to_plot:

            plt.plot(100 - np.array(list(accuracy_for_each_topk_entropy[4].keys())),
                 np.array(list(accuracy_for_each_topk_entropy[4].values())),
                 label=f"Top-{20} Accuracy via Entropy", c="pink", marker='*', markersize=markersize, linewidth=linewidth, linestyle = ':');



    if 1 in top_k_to_plot:
        accuracy_on_full_data_top_1 = np.array(list(accuracy_for_each_topk_delta[0].values()))[-1]
    if 2 in top_k_to_plot:
        accuracy_on_full_data_top_2 = np.array(list(accuracy_for_each_topk_delta[1].values()))[-1]
    if 5 in top_k_to_plot:
        accuracy_on_full_data_top_5 = np.array(list(accuracy_for_each_topk_delta[2].values()))[-1]
    if 10 in top_k_to_plot:
        accuracy_on_full_data_top_10 = np.array(list(accuracy_for_each_topk_delta[3].values()))[-1]
    if 20 in top_k_to_plot:
        accuracy_on_full_data_top_20 = np.array(list(accuracy_for_each_topk_delta[4].values()))[-1]

    if 1 in top_k_to_plot:
        plt.annotate("{}".format(np.round(accuracy_on_full_data_top_1, 2)), (0 - 4.5, accuracy_on_full_data_top_1), size = 14);
    if 2 in top_k_to_plot:
        plt.annotate("{}".format(np.round(accuracy_on_full_data_top_2, 2)), (0 - 4.5, accuracy_on_full_data_top_2), size = 14);
    if 5 in top_k_to_plot:
        plt.annotate("{}".format(np.round(accuracy_on_full_data_top_5, 2)), (0 - 4.5, accuracy_on_full_data_top_5), size = 14);
    if 10 in top_k_to_plot:
        plt.annotate("{}".format(np.round(accuracy_on_full_data_top_10, 2)), (0 - 4.5, accuracy_on_full_data_top_10), size = 14);
    if 20 in top_k_to_plot:
        plt.annotate("{}".format(np.round(accuracy_on_full_data_top_20, 2)), (0 - 4.5, accuracy_on_full_data_top_20), size = 14);

    plt.legend(fontsize=16);

    plt.savefig(f"QA_Rejection_curves_based_on_single_model_UE_mes_for_{name_of_dataset}_{model_name}_model.png", dpi = 200)

def area_under_rejection_curve(data, name, topk, num_quantiles, number_of_digids_round = 4):
    """
    This function calculates absolute Area under Rejection curve to 
    estimate overall quality of uncertainty estimation
    """
    max_area = (100 - 100 / num_quantiles) * 100
    index_of_elem = data["top_ks"].index(topk)

    x = (100.0 - np.array(list(data[f"{name}"][index_of_elem].keys())).astype(float))[::-1]
    y = np.array(list(data[f"{name}"][index_of_elem].values())).astype(float)[::-1]
    area = np.round(trapz(y = y, x = x, dx=1) / max_area, number_of_digids_round)

    return area


    
def count_AURC_sinle_model_UE(data, num_quantiles):
    
    name_of_dataset = data["name_of_dataset"]
    
    names = [
        "accuracy_for_each_topk_entropy",
        "accuracy_for_each_topk_maxprob",
        "accuracy_for_each_topk_delta"
    ]    

    X = [[area_under_rejection_curve(data = data, name = name, topk=topk, num_quantiles = num_quantiles) for topk in data["top_ks"]] for name in names]
    
    df = pd.DataFrame(X)
    df["UE"] = ["entropy", "score", "delta"]
    df.set_index("UE", inplace=True)
    df.columns = [f"{name_of_dataset} top-1 acc",
                  f"{name_of_dataset} top-2 acc",
                  f"{name_of_dataset} top-5 acc",
                  f"{name_of_dataset} top-10 acc",
                  f"{name_of_dataset} top-20 acc"]
    mux = pd.MultiIndex.from_product([[f"{name_of_dataset}"], ['top-1 acc', 'top-2 acc', 'top-5 acc', 'top-10 acc', 'top-20 acc']])

    dta = df.reset_index()[[f"{name_of_dataset} top-1 acc", f"{name_of_dataset} top-2 acc",
                            f"{name_of_dataset} top-5 acc", f"{name_of_dataset} top-10 acc",
                            f"{name_of_dataset} top-20 acc"]]

    new_dta = pd.DataFrame(np.array(dta), columns=mux)
    new_dta["Model"] = ["Single"]*3
    beam_size = data["num_beams"]
    new_dta["Beam Size"] = beam_size
    new_dta["UE"] = ["entropy", "score", "delta"]


    new_dta = new_dta[["Model", "Beam Size", "UE", f"{name_of_dataset}"]]
    new_dta.to_csv(f"AURC_{name_of_dataset}_single_model_bs_{beam_size}.csv")
    
    return new_dta




