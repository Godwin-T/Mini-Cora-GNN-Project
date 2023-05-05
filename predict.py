import torch
import numpy as np
import pandas as pd
import pickle
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

model = torch.load('./model.pth')
with open('data.pkl', 'rb') as f:
    data, class_idc = pickle.load(f)

paper_path = './cora.content'
column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers_data = pd.read_csv(paper_path, 
                            sep="\t", 
                            header=None, 
                            names=column_names,)
papers_data = papers_data.sort_values('paper_id', ascending=True)

def infrence(papers_data):
    ''''For making infrence on new data'''
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

def get_keys(d, value):
    
    for k, v in d.items():
        if v == value:
            return k

model.eval()
node_features, test_data = infrence(papers_data)

new_lab = []
for i in range(node_features.shape[0]):
    
    paper_index = test_data.iloc[0]['paper_id']
    new_data_feats = node_features[i].unsqueeze(0)
    x = torch.cat((data.x, new_data_feats), dim = 0)
    out = model(x,data.edge_index)
    pred = out.argmax(dim=1)  
    new_lab.append(pred[-1])
    print(f'The paper with index id {paper_index} is classified as a {get_keys(class_idc, pred[-1])} paper')



# lab = test_data['subject'].values.tolist()
# new_lab, lab = np.array(new_lab), np.array(lab)
# corr = (new_lab == lab).sum()
# corr/node_features.shape[0]

