import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class POIRecommender(pl.LightningModule):
    def __init__(self, num_users, num_pois, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.poi_embeddings = nn.Embedding(num_pois, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_pois)

    def forward(self, user_ids, poi_ids):
        user_embeddings = self.user_embeddings(user_ids)
        poi_embeddings = self.poi_embeddings(poi_ids)
        x = torch.cat([user_embeddings, poi_embeddings], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.output(x)
        return output

    def training_step(self, batch, batch_idx):
        user_ids, poi_ids, labels = batch
        logits = self(user_ids, poi_ids)
        loss = F.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer