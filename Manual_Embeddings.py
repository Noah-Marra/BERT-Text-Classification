import DB
import Initialization
import UMAP_HDBSCAN
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hdbscan
import umap

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from tqdm.notebook import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from transformers import BertModel, BertTokenizer
from bertopic import BERTopic

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#Load Data
sql_where = " WHERE IsValid=True AND QuestionID=1"
db_path = 'data/'
labelset_id = Initialization.init_labels(db_path)
text = DB.load_data(db_path + 'ConstructMapping.db', sql_where)
#text = DB.pre_process(text)
#--------------------------------------------------------------------------------------------------------
#Load Embedders
#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
use = hub.load(module_url)
#Sentence BERT
sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#BERT-Uncased
class bert_model(nn.Module):
    def __init__(self, model_name, freeze = False, device = 'cpu'):
        super().__init__()

        self.model = BertModel.from_pretrained(model_name)
        self.device = device

        if freeze:
            for layer in self.model.parameters():
                layer.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        # Obtain BERT embeddings
        with torch.no_grad():
            outputs = self.model(x['input_ids'])
            embeddings = outputs.last_hidden_state
            mean_pool = embeddings.sum(axis=1)

        return mean_pool
model_BERT = 'bert-base-uncased'
bert = bert_model(model_BERT, freeze=True)
#--------------------------------------------------------------------------------------------------------
#dataloader
def embed(model, model_type, text):
    final_embeddings=list()
    all_embeddings = []
    final_sentences = text

    batch_sz = 200 # batch_size
    for i in range(0, len(final_sentences), batch_sz):
        batch_sentences = final_sentences[i:i+batch_sz]
        for sent in batch_sentences:

            if model_type == 'use':
                tokens = tokenizer(sent, return_tensors='pt')
                embeddings = model(tokens)
                all_embeddings.extend(embeddings)

            elif model_type == 'bert':
                tokens = tokenizer(sent, return_tensors='pt')
                embeddings = model(tokens)
                final_embeddings.extend(embeddings)
                all_embeddings = torch.stack(final_embeddings)

            elif model_type == 'sbert':
                embeddings = model.encode(sent)
                all_embeddings.append(embeddings)

    return all_embeddings

#--------------------------------------------------------------------------------------------------------
#Choose your Embeddings
#embeddings = embed(use, 'use', text)
#embeddings = embed(bert, 'bert', text)
embeddings = embed(sbert,'sbert', text)
#--------------------------------------------------------------------------------------------------------
#Choose your hyperparameters
min_cluster_size = 11
n_components = 12
n_neighbors = 22
random_state = 15
min_samples = None

#--------------------------------------------------------------------------------------------------------
umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,n_components=n_components,metric='cosine',
                            random_state=random_state).fit_transform(embeddings))

clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,metric='euclidean',
                               gen_min_span_tree=True, cluster_selection_method= 'eom').fit(umap_embeddings)


probabilities = clusters.probabilities_
labels = clusters.labels_
print(f'The number of labels is: {len(np.unique(clusters.labels_))}')
#--------------------------------------------------------------------------------------------------------
#Export Data
DB.export_data(labels, labelset_id, probabilities, db_path + 'ConstructMapping.db', sql_where)
#--------------------------------------------------------------------------------------------------------
#combined_strings = c_TF_IDF.cluster_combiner(labels, text)
#tf_idf, count = c_TF_IDF.c_tf_idf(combined_strings, len(np.unique(clusters.labels_)), ngram_range=(1,1))
