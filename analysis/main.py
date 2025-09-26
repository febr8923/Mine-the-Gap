import pandas as pd
import torch
from torch_geometric.data import TemporalData
from torch_geometric.nn.models import TGN

dataset = pd.read_csv('./data/edge_events.csv')

# Extract unique nodes and create mapping dictionaries
nodes = pd.unique(dataset[["src", "dst"]].to_numpy().ravel())
node2idx = {n: i for i, n in enumerate(nodes)}
idx2node = {i: n for n, i in node2idx.items()}

# Map source and destination nodes to integer IDs
src = dataset["src"].map(node2idx).to_numpy()
dst = dataset["dst"].map(node2idx).to_numpy()

# Extract timestamps and labels
t = dataset["timestamp"].astype(int).to_numpy()
msg = pd.get_dummies(dataset["label"]).to_numpy()

# Build TemporalData object
data = TemporalData(
    src=torch.from_numpy(src).long(),
    dst=torch.from_numpy(dst).long(),
    t=torch.from_numpy(t).long(),
    msg=torch.from_numpy(msg).float(),
)

tgn_model = TGN(num_nodes=len(nodes), embedding_dim=64)
node_embeddings = tgn_model(data)

edge_scores = (node_embeddings[src] * node_embeddings[dst]).sum(dim=-1)