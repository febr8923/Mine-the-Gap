import pandas as pd
import torch
from torch_geometric_temporal.data import TemporalData
from torch_geometric_temporal.nn.recurrent import TGN
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# Example data loading (user should replace with actual dataset loading code)
# dataset = pd.read_csv("your_dataset.csv")

# Assuming dataset has columns: src, dst, timestamp, label
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

# Build TemporalData object
data = TemporalData(
    src=torch.from_numpy(src).long(),
    dst=torch.from_numpy(dst).long(),
    t=torch.from_numpy(t).long(),
    msg=torch.from_numpy(msg).float(),
)

# Define Temporal Graph Network (TGN) model
class TGNModel(torch.nn.Module):
    def __init__(self, node_count, memory_dim=32, message_dim=32, embedding_dim=32):
        super(TGNModel, self).__init__()
        self.tgn = TGN(node_count, memory_dim, message_dim, embedding_dim)
        self.lin = torch.nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, src, dst, t):
        # Get node embeddings from TGN
        x_src, x_dst = self.tgn(src, dst, t)
        # Compute embedding difference as feature
        emb_diff = torch.abs(x_src - x_dst)
        out = self.lin(emb_diff)
        return out

# Initialize model
num_nodes = len(nodes)
model = TGNModel(num_nodes)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop for unsupervised anomaly detection
def train(data, model, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.src, data.dst, data.t)
        # Use simple reconstruction loss (example)
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

train(data, model)

# After training, get anomaly scores as embedding distances or reconstruction errors
model.eval()
with torch.no_grad():
    embeddings = model(data.src, data.dst, data.t)
    anomaly_scores = torch.norm(embeddings, dim=1).cpu().numpy()

# Output anomaly scores can be used for further analysis or flagged for investigation
