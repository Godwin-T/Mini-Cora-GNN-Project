import torch
import numpy as np
import pandas as pd
import pickle
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


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

def batch_prediction(model, graph_data, citation_data,  data):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    feature_names = set(data.columns) - {"paper_id", "subject"}
    node_features = data[feature_names].to_numpy()
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

    new_lab = []
    for i in range(node_features.shape[0]):

        new_data_feats = node_features[i].unsqueeze(0)

        target_ = citation_data[citation_data['target']==i] 
        edge = torch.from_numpy(target_.to_numpy()).T

        x = torch.cat((graph_data.x, new_data_feats), dim = 0)
        new_graph_edge = torch.cat((graph_data.edge_index, edge), dim = 1)

        new_graph_data = Data(x= x, edge_index = new_graph_edge)
        new_graph_data = new_graph_data.to(device)

        #out = model(x,new_graph_edge)
        out = model(new_graph_data.x, new_graph_data.edge_index)
        pred = out.argmax(dim=1)  
        new_lab.append(pred[-1])
    return new_lab

def single_prediction(model, graph_data, data, new_edges):

    model.eval()
    
    node_features = torch.from_numpy(data).type(torch.FloatTensor)
    node_features = node_features.unsqueeze(0)

    x = torch.cat((graph_data.x, node_features), dim = 0)
    new_edge = torch.cat((graph_data.edge_index, new_edges), dim = 1)

    out = model(x,new_edge)
    pred = out.argmax(dim=1)[-1]
    return pred


citation_path = './cora.cites'
paper_path = './cora.content'

model = torch.load('model.pth')
with open('data.pkl', 'rb') as f:
    graph_data = pickle.load(f)

papers_data, citations_data = data_prep(citation_path, paper_path)
papers_data, citations_data, mappings = value_mapping(papers_data, citations_data)

out = batch_prediction(model, graph_data, citations_data, papers_data)
lab = papers_data['subject'].values.tolist()
new_lab, lab = np.array(out), np.array(lab)
corr = (new_lab == lab).sum()
print('Prediction Acc' ,corr/papers_data.shape[0])

ori = np.array(graph_data.y)
corr = (new_lab == ori).sum()
print('Actual Acc' ,corr/papers_data.shape[0])