# Uncertainty Enhances Knowledge Base Question Answering

Questions Answering (QA) is one of the most
widespread tasks in Natural Language Process-
ing (NLP). Knowledge Base (KB) is one of the
approaches that helps to extract answers more ac-
curately. This work focuses on the application of
uncertainty estimation to extend the capabilities
of the KBQA system. Using uncertainty estima-
tion techniques, it is possible to sort answers by
confidence and filter those questions for which
the model is less confident. It is demonstrated
that uncertainty can enhance the performance of
end-to-end KBQA system and a particular module
of such system – Entity Linker. Experiments in-
cluded various uncertainty estimation approaches
based on single model and ensemble estimations
on different QA datasets. It is also shown that
uncertainty estimates can be used to study the
behavior of the model when answering different
types of questions.

All theory and results are discribed in `document`.

This repository will be devided on 2 parts:
1) End-to-end KBQA model based on T5.
2) Multilingual EL as a sufficient module of QA system based on mGENRE.

# KBQA via T5


This part contains Uncertainty Estimation expiriments with T5 model - end-to-end QA model.
We follow the article https://arxiv.org/abs/2002.08910 and use T5 as a base model for question answering task. It operates with KB and works as a seq-to-seq model.

## Experiments with Uncertainty estimations with T5.

We conducted experiments with several single unceratinty measures: `Entropy`, `Maxprob`, `Delta`. And we also conducted experiments with MC-Dropout-based ensemble metrics: `Ensemble score`, `Ensemble delta`, `Expected entropy`, `Predictive entropy`, `BALD`, `EPKL`, `RMI`. 

## Quickstart T5

0) Clone repository

```bash
git clone https://github.com/SergeyPetrakov/Master_thesis
cd Master_thesis
```

1) data copy command

```cp -r data/. .```

2) Choose any way of reproducing

Since T5 model was taken from Hugging Face one can easily launched.

There are two ways to reproduce results: 

1) ```Jupyter notebooks```
"Run all" notebooks for uncertainty experiments with all sufficient information are available. 
For single models: ```CLEAR_SINGLE_MODEL_UE_T5.ipynb```
For ensemble models: ```CLEAR_MC_DROPOUT_UE_T5.ipynb```

Some additional experiments available in jupyter notebooks, that starts with `QA`.


2) ```Python scripts```
Since some operations require significant amount of time even with GPU file there are added `t5_ue.py` and `t5_ue_with_trie.py` in order to launch them using `nohup python3 t5_ue.py > t5_ue.out &`. After that all necessary tables and figures will be saved automatically in working folder.

Note: trie file adds an opporunity to take into account graph structure during T5 generation process.

Some results presented in presentations:
Results of MC dropout model uncertainty experiments ( Ensemble score, Ensemble delta, Expected entropy, Predictive entropy, BALD, EPKL, RMI) are available in `mc_dropout_experiments_uncertainty.pptx`
Results of sinlge model uncertainty experiments (Maxprob, Delta, Entropy) are available in `16_11_2022_baseline_acc_single_uncertainty.pptx`.

## Results

### Summary of results on UE T5 part.

1) Single model-based metrics often demonstrate good performance. They outperform ensemble metrics on RuBQ 2.0 dataset and SQ in graph part.
2) Almost each time score is the best Single model-based metric from Single model-based metrics.
3) Ensemdle-based models outperform single-based models only on SQ full test dataset. 

For Simple Questions we observe the following results:
- Expected entropy – worse behavior on high rejection rates
- Predictive entropy, EPKL, BALD stable and very similar
- Best performance (especially for low rejection rates) for Delta, Maxprob and Eentropy. (All of them perform worse on high rejection rates)

For RuBQ 2.0 we observe the following results:
- Expected entropy – worse behavior on high rejection rates
- Predictive entropy, EPKL, BALD stable and very similar, but gain is small.
- Best performance by Delta, Maxprob
- Eentropy perform very well at low rejection rates
- Model fails on questions w/o answer

5) Gain from using trie during hypothesis generation is very negligable.
6) Changing dropout rate reduces results.





# Multilingual Entity Linking via mGENRE

This part contains files and materials related to multilingual entity linking task (MEL), especially basing on the mGENRE model since it is SOTA model. We consider MEL as a part of big knowledge base question answering (KBQA) that is called information retrieval part. Within this part we retrieve entities. Basing on them we can make queries to knowledge base. Thus, we obtain KBQA system.
This part is based on https://arxiv.org/abs/2103.12528 where mGENRE model is proposed.


## Quickstart mGENRE

Highly recommend to launch mGENRE using docker:

0) Clone repository

```bash
git clone https://github.com/SergeyPetrakov/Master_thesis
cd Master_thesis
```

1) Run docker container

```bash
docker build -f ./Dockerfile -t ue_dev ./
docker run -v $PWD:/mel_ue/ --network host -ti ue_dev
```

2) You hae to run the code below to be able to do useful manipulations

```bash
apt update
apt install vim -y
apt install git -y
apt install wget -y
```

3) You have to move GENRE, KILT and fairseq into mel_ue directory

```bash
mv GENRE mel_ue/
mv KILT mel_ue/
mv fairseq mel_ue/
```

4) It is necessary to add fairseq_model.py 

```bash
cd mel_ue
mv fairseq_model.py fairseq/fairseq/models
```

5) Now you can run `mGENRE_and_UE.py` in order to reproduce some results (There is one particular experiment, but one can easily switch to data you want, or parameters you want in the last part of code)

```bash
python3 mGENRE_and_UE.py
```
---------------------------------------------------------------------------
Notes:
If you are going to use GPU it is worth run additionally the following command (to overcome difficulties with old versions of cuda and torch)
```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

---------------------------------------------------------------------------

Approach presented below works unstable but it is a good idea to illustrate alternatives of docker launching.
In command line run the folllowing six commands:

1) set new environment `conda create --name new_environment`
2) activate new environment `conda activate new_environment`
3) clone repository `git clone https://github.com/SergeyPetrakov/Bayesian_ML_project`
4) go to the cloned repository `cd Bayesian_ML_project`
5) install requirements `pip install -r requirements.txt`
6) launch jupyter notebook (for example: `jupyter notebook --ip 0.0.0.0 --port=7643 --no-browser --allow-root&`)
and open mgenre_final there (paste in browser web link and open file)

Necessary data and pretrained model jupyter notebook ```mGENRE_and_Uncertainty_Estimation.ipynb```contain in cell `data`. If you once installed it you do not really need run this cell further.
We strongly recommend to follow the original article and repository to understand how everything works from the inside.

## Experiments with Uncertainty estimations with mGENRE

Experiments are provided in `mGENRE_and_Uncertainty_Estimation.ipynb` file, you can find there:
 - Quickstart
 - Experiments with uncertainty estimation using metrics: `Entropy` on a single model, `Delta`, `Maxprob`, `Predictive entropy`, `Expected entropy` and `BALD` on `Simple Questions`, `RuBQ 2.0` and `Mewsli-9` datasets
 
File `mGENRE_and_UE.py` is also provided (repeat results from jupyter notebook `mGENRE_and_Uncertainty_Estimation.ipynb`). The advantage of .py files over jupyter notebooks is that one can launch them for example on the server and be sure in stability of connection for the whole period of experiment that may take hours.
 
 If you want to launch `mGENRE_and_UE.py` script on the remote server running after closing out of SSH you should run `nohup python3 mGENRE_and_UE.py &`
 
 ## Results
 
 ### Illustrations and tables

File `3_dataset_experiments.pdf` contains rejection curves - visual illustration of uncertainty estimation integration into mGENRE quality assessment. As a numerical measure of unsertainty estimation quality were added two types of ares under rejection curve: absolute (equals to the whole area under curve) and comparative (equals to the area that is higher than quality received on all samples of dataset).

### Summary of results on Multilingual Entity Linking part

- Experiments demonstrate that uncertainty quantification could be efficiently used for the task of entity linking in case of mGENRE model. This is especially obvious for English, German, Russian. Although there are cases when uncertainty estimation does not help, for example for Serbian and in some cases for Javanese, Japanese and Persian.
- The structure of dataset also matters, because the structure differs, for example model may perform better on RUBQ 2.0 and Simple Questions because they consist of short questions of widespread languages (English and Russian).
- There is no significant leader in terms of metric of uncertainty, but every time both for absolute area under rejection curve and area under curve added by uncertainty integration predicted entropy showed the best results among all metrics almost each time as for information received on aggregated AUC data.
- Thus, mGENRE performs well on the task of entity linking even in multilingual case. But end-to-end question answering system could not be realized using only mGENRE, even if we help it with such strong NER as Stanza.

### Conclusion on Multilingual Entity Linking part
I can say that reached initial objectives. I conducted many experiments on entity linking and question answering task for mGENRE model, observed different datasets, types of questions (with single answer or potentially not one answer), checked different languages, quantified uncertainty using various metrics, such as entropy, maxprob, delta, predictive entropy, expected entropy and BALD. Provided quantifiable and comparable results using area under rejective curve approach.
Talking about the ways how one can take benefits from it, I can say that method of voting algorithms can be used, when we take some algorithms, mark answers they give with some uncertainty measure and choose the answer of the most confident algorithm among all.
