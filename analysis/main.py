import pandas as pd
import torch
from torch_geometric.data import TemporalData
from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn.functional as F

# Example dataset loading (replace with actual)
dataset = pd.read_csv("data/edge_events_clean.csv")

# Extract unique nodes and create mapping dictionaries
nodes = pd.unique(dataset[["src", "dst"]].to_numpy().ravel())
node2idx = {n: i for i, n in enumerate(nodes)}
idx2node = {i: n for n, i in node2idx.items()}

# Map source and destination nodes to integer IDs
src = dataset["src"].map(node2idx).to_numpy()
dst = dataset["dst"].map(node2idx).to_numpy()

# Extract timestamps and one-hot encode labels
t = dataset["timestamp"].astype(int).to_numpy()
msg = pd.get_dummies(dataset["label"]).to_numpy()

# Note: TGCN expects node features and edge_index per time step.
# We will group data by timestamp slices for TemporalData

# Organize edge indices and features per timestamp
max_time = t.max()
edge_indices = []
features = []

for time in range(max_time + 1):
    mask = (t == time)
    if mask.sum() == 0:
        # If no edges at this time, add empty tensor (handle carefully in training)
        edge_index = torch.empty((2,0), dtype=torch.long)
        feat = torch.empty((len(nodes), msg.shape[1]), dtype=torch.float)
        feat[:] = 0
    else:
        # Build edge_index tensor (2 x num_edges)
        e_src = torch.tensor(src[mask], dtype=torch.long)
        e_dst = torch.tensor(dst[mask], dtype=torch.long)
        edge_index = torch.stack([e_src, e_dst], dim=0)
        # Initialize node features as zeros except for edges involved (simple example)
        feat = torch.zeros((len(nodes), msg.shape[1]), dtype=torch.float)
        for s, label_vec in zip(e_src, torch.tensor(msg[mask], dtype=torch.float)):
            feat[s] = label_vec
    edge_indices.append(edge_index)
    features.append(feat)

# Create TemporalData object with slices of features and edge indices
data = TemporalData(x=features, edge_index=edge_indices)

# Define TGCN model
class TGCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TGCNModel, self).__init__()
        self.tgcn = TGCN(in_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)  # Output anomaly score per node

    def forward(self, x, edge_index):
        h = self.tgcn(x, edge_index)
        out = self.lin(h)
        return out.squeeze()

# Initialize model
in_channels = msg.shape[1]
hidden_channels = 32
model = TGCNModel(in_channels, hidden_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training example for unsupervised anomaly detection
def train(data, model, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        # TemporalData x and edge_index are lists of time steps
        losses = []
        for x_t, edge_index_t in zip(data.x, data.edge_index):
            if edge_index_t.numel() == 0:
                # Skip empty times to avoid error
                continue
            out = model(x_t, edge_index_t)
            loss = F.mse_loss(out, torch.zeros_like(out))
            losses.append(loss)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

train(data, model)

# To compute anomaly scores (example: absolute output as anomaly score per node)
model.eval()
anomaly_scores = []
with torch.no_grad():
    for x_t, edge_index_t in zip(data.x, data.edge_index):
        if edge_index_t.numel() == 0:
            anomaly_scores.append(torch.zeros(len(nodes)))
            continue
        out = model(x_t, edge_index_t)
        scores = out.abs()
        anomaly_scores.append(scores.cpu())

# anomaly_scores is a list of tensors per time step
