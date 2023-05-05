# Importing Libraries
print('Importing Libraries')
import pandas as pd
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings('ignore')

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pickle

# Load data
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

citation_path = './cora.cites'
paper_path = './cora.content'



def data_prep(citation_path, paper_path):
    
    # Loading citation data
    citations_data = pd.read_csv(citation_path,
                                    sep="\t",
                                    header=None,
                                    names=["target", "source"],
                                    )
    # Loading papers data
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers_data = pd.read_csv(paper_path, 
                                sep="\t", 
                                header=None, 
                                names=column_names,)
    papers_data = papers_data.sort_values('paper_id', ascending=True)
    return papers_data, citations_data


def value_mapping(papers_data, citations_data):

    # Class mapping i
    class_values = sorted(papers_data["subject"].unique())
    class_idc = {name: id for id, name in enumerate(class_values)}

    # Paper Id mapping
    paperid_values = sorted(papers_data["paper_id"].unique())
    paper_idc = {name: idx for idx, name in enumerate(paperid_values)}

    papers_data["paper_id"] = papers_data["paper_id"].apply(lambda name: paper_idc[name])
    citations_data["source"] = citations_data["source"].apply(lambda name: paper_idc[name])
    citations_data["target"] = citations_data["target"].apply(lambda name: paper_idc[name])
    papers_data["subject"] = papers_data["subject"].apply(lambda value: class_idc[value])

    mappings = (class_idc, paper_idc)

    return papers_data, citations_data, mappings


def extract_features(papers_data, citations_data):
    
    # get node feature names
    feature_names = set(papers_data.columns) - {"paper_id", "subject"}

    # create edges array [2, num_edges].
    edges = citations_data[["source", "target"]].to_numpy().T
    edge_index = torch.from_numpy(edges).to(torch.long)

    # create node features array [num_nodes, num_features].
    node_features = papers_data.sort_values("paper_id")[feature_names].to_numpy()
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    labels = torch.from_numpy(papers_data["subject"].values).to(torch.long)

    # create graph data
    data = Data(x=node_features, edge_index = edge_index, y=labels)

    return data


# Define model
class GCN(torch.nn.Module):

    def __init__(self, input_feats, hidden_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(input_feats, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)

        # Output layer 
        x = F.log_softmax(self.out(x), dim=1)
        return x



def train(utils, new_data):

      model, optimizer, criterion = utils

      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(new_data.x, new_data.edge_index)
      
      pred = out.argmax(dim=1)  
      test_correct = (pred[data.train_mask] == new_data.y[data.train_mask])
      acc = int(test_correct.sum()) / int(data.train_mask.sum())
     
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], new_data.y[data.train_mask]) 
      loss.backward() 
      optimizer.step()
      return loss, acc

def test(model, new_data):
      model.eval()
      out = model(new_data.x, new_data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = (pred[data.test_mask] == new_data.y[data.test_mask])  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

def val(model, new_data):
      model.eval()
      out = model(new_data.x, new_data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = (pred[data.val_mask] == new_data.y[data.val_mask])  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.val_mask.sum())  
      return test_acc


def get_keys(d, value):
    '''For mapping predicted values to their keys'''
    for k, v in d.items():
        if v == value:
            return k


def new_input(new_node, citation):
    '''For preparing new data'''
    x = torch.cat((new_data.x, new_node), dim = 0)
    context_edges = citation.to_numpy()
    context_edges = torch.from_numpy(context_edges).to(torch.long).T
    x_index = torch.cat((new_data.edge_index, context_edges), dim = 1)
    return x, x_index

print('Loading Data')
papers_data, citations_data = data_prep(citation_path, paper_path)
papers_data, citations_data, mappings = value_mapping(papers_data, citations_data)
print('Feature Extraction')
new_data = extract_features(papers_data, citations_data)
class_idc, paper_idc = mappings


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.crd = CRD(1433, 64, 0.5)
        self.cls = CLS(64, 7)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x

print('Initializing  model Parameters')
# Initialize model
input_feats = new_data.x.shape[1]
hidden_channels = 64
out_channels = 7

model = GCN(input_feats, hidden_channels, out_channels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU
model = model.to(device)
new_data = new_data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)

# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

utils = (model, optimizer, criterion)

print('Model training')
losses = []
for epoch in range(0, 1001):
    loss, acc = train(utils, new_data)
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Accuracy: {acc:.4f}')

print('=========================================================================================')


test_acc = test(model, new_data)
val_acc = val(model, new_data)

print(f'Test Accuracy: {test_acc:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')
torch.save(model, 'model.pth')
with open('data.pkl', 'wb') as f:
    pickle.dump((new_data, class_idc), f)