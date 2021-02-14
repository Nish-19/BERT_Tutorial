# BERT_Tutorial

Code for using BERT embeddings and fine-tuning for classification tasks.

Here, BERT is used on a sentence pair for stance detection on Covid-Stance dataset. The dataset is taken from twitter and the code to rehydrate the tweets is given in the dataset folder.

bert_embeddings.py - uses BERT based embeddings for the stance detection task.
bert_fine_tune.py - Performs Fine-Tuning using the BertForSequenceClassification module of HuggingFace Transformers.
