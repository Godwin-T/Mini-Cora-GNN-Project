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

dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

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
    st_edges = citations_data[["source", "target"]].to_numpy().T
    ts_edges = citations_data[["target", "source"]].to_numpy().T
    edges = np.concatenate([st_edges, ts_edges], axis=1)
    edge_index = torch.from_numpy(edges).to(torch.long)

    # create node features array [num_nodes, num_features].
    node_features = papers_data.sort_values("paper_id")[feature_names].to_numpy()
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    labels = torch.from_numpy(papers_data["subject"].values).to(torch.long)

    # create graph data
    data = Data(x=node_features, edge_index = edge_index, y=labels)

    # print("Edges shape:", edges.shape)
    #print("Nodes shape:", node_features.shape)
    return data

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
    
    for k, v in d.items():
        if v == value:
            return k

def new_input(new_node, citation, new_data, ):

    x = torch.cat((new_data.x, new_node), dim = 0)
    context_edges = citation.to_numpy()
    context_edges = torch.from_numpy(new_citation).to(torch.long).T
    x_index = torch.cat((new_data.edge_index, context_edges), dim = 1)
    return x, x_index

def infrence(papers_data):
    train_data, test_data = [], []
    for _, group in papers_data.groupby("subject"):
        # Select around 50% of the dataset for training.
        random_selection = np.random.rand(len(group.index)) <= 0.8
        train_data.append(group[random_selection])
        test_data.append(group[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    # get node feature names
    feature_names = set(papers_data.columns) - {"paper_id", "subject"}

    # create node features array [num_nodes, num_features].
    node_features = test_data[feature_names].to_numpy()
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    return node_features,test_data

citation_path = './cora.cites'
paper_path = './cora.content'
# Initialize model
input_feats = 1433 #new_data.x.shape[1]
out_channels = 7

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4

def final_train(citation_path, paper_path, 
                input_feats = 1433, hidden_channels = 16,
                out_channels = 7, learning_rate = 0.01,
                decay = 5e-4, epochs = 1001):

    papers_data, citations_data = data_prep(citation_path, paper_path)
    papers_data, citations_data, mappings = value_mapping(papers_data, citations_data)
    new_data = extract_features(papers_data, citations_data)
    class_idc, paper_idc = mappings

    

    model = GCN(input_feats, hidden_channels, out_channels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU
    model = model.to(device)
    new_data = new_data.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=decay)

    # Define loss function (CrossEntropyLoss for Classification Problems with 
    # probability distributions)
    criterion = torch.nn.CrossEntropyLoss()
    utils = (model, optimizer, criterion)

    losses = []
    for epoch in range(0, epochs):
        loss, acc = train(utils, new_data)
        losses.append(loss)
        #if epoch % 100 == 0:
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Accuracy: {acc:.4f}')

    test_acc = test(model, new_data)
    print(f'Test Accuracy: {test_acc:.4f}')

    val_acc = val(model, new_data)
    print(f'Validation Accuracy: {val_acc:.4f}')
    return model, papers_data, citations_data, new_data, class_idc

model, papers_data, citation_data, new_data, class_idc = final_train(citation_path, paper_path)

feature_names = set(papers_data.columns) - {"paper_id", "subject"}
lab = papers_data['subject'].values.tolist()

# create node features array [num_nodes, num_features].
node_features = papers_data[feature_names].to_numpy()
node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

torch.save(model, 'model.pth')
print('Model Saved')

model.eval()

new_lab = []
for i in range(node_features.shape[0]):
    
    new_data_feats = node_features[i].unsqueeze(0)

    target_ = citation_data[citation_data['target']==i] 
    edge = torch.from_numpy(target_.to_numpy()).T

    x = torch.cat((new_data.x, new_data_feats), dim = 0)
    new_edge = torch.cat((new_data.edge_index, edge), dim = 1)

    out = model(x,new_edge)
    pred = out.argmax(dim=1)  
    new_lab.append(pred[-1])


new_lab, lab = np.array(new_lab), np.array(lab)
corr = (new_lab == lab).sum()
print(corr/node_features.shape[0])

def batch_prediction(model, graph_data, citation_data,  data):


    model.eval()

    feature_names = set(data.columns) - {"paper_id", "subject"}

    # create node features array [num_nodes, num_features].
    node_features = data[feature_names].to_numpy()
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

    new_lab = []
    for i in range(node_features.shape[0]):

        new_data_feats = node_features[i].unsqueeze(0)

        target_ = citation_data[citation_data['target']==i] 
        edge = torch.from_numpy(target_.to_numpy()).T

        x = torch.cat((graph_data.x, new_data_feats), dim = 0)
        new_edge = torch.cat((graph_data.edge_index, edge), dim = 1)

        out = model(x,new_edge)
        pred = out.argmax(dim=1)  
        new_lab.append(pred[-1])
    return new_lab

citation_path = './cora.cites'
paper_path = './cora.content'

model = torch.load('./model.pth')

papers_data, citations_data = data_prep(citation_path, paper_path)
papers_data, citations_data, mappings = value_mapping(papers_data, citations_data)

out = batch_prediction(model, new_data, citations_data, papers_data,)
lab = papers_data['subject'].values.tolist()
new_lab, lab = np.array(out), np.array(lab)
corr = (new_lab == lab).sum()
print(corr/papers_data.shape[0])
