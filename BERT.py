import sqlite3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import BertModel, BertTokenizer

#--------------------------------------------------------------------------------------------------------

#Load Data
conn = sqlite3.connect('C:/Users/nmarr/OneDrive/Documents/Benchcube/Data/ConstructMapping.db')
cursor = conn.cursor()
cursor.execute("SELECT Response FROM Response")

rows = cursor.fetchall()
text = []
for row in rows:
    row = str(row)
    cleaned = row[2:-3]
    text.append(cleaned)
print(text)

#--------------------------------------------------------------------------------------------------------

#create class for model
class model(nn.Module):
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

#--------------------------------------------------------------------------------------------------------

#load model
model_name = 'bert-base-uncased'
bert = model(model_name, freeze=True)
tokenizer = BertTokenizer.from_pretrained(model_name)

#--------------------------------------------------------------------------------------------------------

#dataloader
final_embeddings=list()
all_embeddings = []
final_sentences = text

batch_sz = 200 # batch_size
for i in range(0, len(final_sentences), batch_sz):
    batch_sentences = final_sentences[i:i+batch_sz]
    for sent in batch_sentences:
        tokens = tokenizer(sent ,truncation='longest_first', return_tensors='pt', return_attention_mask=True,padding=True)
        embeddings = bert(tokens)
        final_embeddings.extend(embeddings)
        all_embeddings = torch.stack(final_embeddings)

#--------------------------------------------------------------------------------------------------------

#Cluster Algorithm
K_range = range(2, 30)
inertia_values = []
best_score = -1

for k in K_range:
    clusterer = KMeans(n_clusters=k, verbose=True)
    clustered_text = clusterer.fit_predict(all_embeddings.cpu())

    inertia_values.append(clusterer.inertia_)
    score.append(silhouette_score(all_embeddings, clustered_text))

#--------------------------------------------------------------------------------------------------------
#Identify the Silhouette Score for Optimal K-Value
max = argmax(score)
cluster_score = score(max)
optimal_K = max + 2

#--------------------------------------------------------------------------------------------------------
#Identify the Elbow Point for Optimal K-Value
"""elbow_point = 0
for i in range(1, len(inertia_values)-1):
    if inertia_values[i] - inertia_values[i+1] < 0.2* (inertia_values[i-1] - inertia_values[i]):
        elbow_point = i + 1
        break
"""

#--------------------------------------------------------------------------------------------------------
print(f"The Optimal Number of Clusters is: {optimal_K}")

#Visualize Inertia Values
plt.plot(K_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()

clusterer_optimal = KMeans(n_clusters=optimal_K, verbose=True)
clustered_text_optimal = clusterer_optimal.fit_predict(all_embeddings.cpu())
labels = clusterer_optimal.labels_

#--------------------------------------------------------------------------------------------------------

#Graphical Representation
print(labels)

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D
ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], all_embeddings[:, 2], c=labels, cmap='viridis')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()











