# BERT_Tutorial

This code is written as a supplement to the blog [BERTology Transfer Learning in Natural Language Processing](https://nish-19.github.io/BERTology-Transfer-Learning-in-Natural-Language-Processing/) written on my website.

Here, BERT is used on a sentence pair for stance detection on Covid-Stance dataset. The dataset is taken from twitter and the code to rehydrate the tweets is given in the dataset folder.

**bert_embeddings.py** - uses BERT based embeddings for the stance detection task.   
**bert_fine_tune.py** - Performs Fine-Tuning using the BertForSequenceClassification module of HuggingFace Transformers.

Before running the above codes, rehydrate the Covid-stance dataset using
cd datasets
python info_extractor.py
python split_traindata.py
