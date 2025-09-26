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

# --- Testing on "not clean" data ---

# 1. Load the unclean dataset
# Assuming the unclean data is in a file like this.
# This file should contain the edges you want to test for anomalies.
try:
    test_dataset = pd.read_csv("data/edge_events.csv")
except FileNotFoundError:
    print("Skipping test phase: 'data/edge_events.csv' not found.")
    test_dataset = None

if test_dataset is not None:
    # Map nodes to IDs using the same mapping from training (no preprocessing)
    # Handle cases where nodes don't exist in training data
    test_src = test_dataset["src"].map(node2idx)
    test_dst = test_dataset["dst"].map(node2idx)
    
    # Fill NaN values with -1 to indicate unknown nodes
    test_src = test_src.fillna(-1).astype(int).to_numpy()
    test_dst = test_dst.fillna(-1).astype(int).to_numpy()
    test_t = test_dataset["timestamp"].astype(int).to_numpy()

    # Check if we have any test data left after filtering
    if len(test_dataset) == 0:
        print("No test edges remain after filtering out unknown nodes.")
    else:
        # 3. Calculate anomaly scores for each edge in the test set
        model.eval()
        edge_anomaly_scores = []
        with torch.no_grad():
            # Get node anomaly scores for all relevant timestamps
            node_scores_per_t = {}
            max_test_time = test_t.max()
            for t_step in range(max_test_time + 1):
                if t_step < len(anomaly_scores):
                    # Use pre-computed scores from the training data timeline if available
                    node_scores_per_t[t_step] = anomaly_scores[t_step]
                else:
                    # If test data has timestamps beyond training data, we'd need to re-run or handle
                    # For simplicity, we assume test timestamps are within the trained range
                    node_scores_per_t[t_step] = torch.zeros(len(nodes)) # Default to zero score

            # Calculate score for each edge by combining node scores
            for i in range(len(test_dataset)):
                t_val = test_t[i]
                src_id = test_src[i]
                dst_id = test_dst[i]

                # Get node scores at the specific timestamp of the edge
                scores_at_t = node_scores_per_t.get(t_val, torch.zeros(len(nodes)))
                
                # Handle unknown nodes (ID = -1)
                src_score = scores_at_t[src_id].item() if src_id >= 0 else 0.0
                dst_score = scores_at_t[dst_id].item() if dst_id >= 0 else 0.0

                # Combine node scores to get an edge score (e.g., max or sum)
                edge_score = max(src_score, dst_score)
                edge_anomaly_scores.append(edge_score)

        test_dataset['anomaly_score'] = edge_anomaly_scores

        # 4. Identify anomalies based on a threshold
        # Example: set threshold as the 95th percentile of scores
        if not test_dataset.empty:
            score_threshold = test_dataset['anomaly_score'].quantile(0.95)
            anomalous_edges = test_dataset[test_dataset['anomaly_score'] > score_threshold]

            print("\n--- Anomaly Detection Results on Unclean Data ---")
            print(f"Score threshold (95th percentile): {score_threshold:.4f}")
            print(f"Found {len(anomalous_edges)} potential anomalous edges.")
            if not anomalous_edges.empty:
                print("Top 5 most anomalous edges:")
                print(anomalous_edges.sort_values('anomaly_score', ascending=False).head())# anomaly_scores is a list of tensors per time step
