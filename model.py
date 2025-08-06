import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ModalityEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.encoder(x)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(AttentionFusion, self).__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.omega = nn.Linear(attention_dim, 1)

    def forward(self, modalities):
        # modalities: List of [batch, hidden_dim, H, W]
        vecs = [m.mean(dim=[2, 3]) for m in modalities]  # Global Average Pooling
        vecs_stack = torch.stack(vecs, dim=1)  # [B, M, D]

        h = torch.tanh(self.W(vecs_stack))  # [B, M, A]
        attn = torch.softmax(self.omega(h), dim=1)  # [B, M, 1]
        fused = (attn * vecs_stack).sum(dim=1)  # [B, D]
        return fused

class HierarchicalReasoning(nn.Module):
    def __init__(self, input_dim, num_levels=3):
        super(HierarchicalReasoning, self).__init__()
        self.levels = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_levels)
        ])

    def forward(self, x):
        embeddings = []
        for layer in self.levels:
            x = torch.relu(layer(x))
            embeddings.append(x)
        return embeddings  # List of hierarchical embeddings

class MedForm(nn.Module):
    def __init__(self, input_channels_dict, hidden_dim=64, attention_dim=32, num_classes=2):
        super(MedForm, self).__init__()
        self.modalities = list(input_channels_dict.keys())
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(input_channels_dict[m], hidden_dim) for m in self.modalities
        })
        self.fusion = AttentionFusion(hidden_dim, attention_dim)
        self.hierarchy = HierarchicalReasoning(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs, masks=None):
        encoded = []
        for m in self.modalities:
            if masks is not None and not masks[m]:
                encoded.append(torch.zeros_like(self.encoders[m](inputs[m])))
            else:
                encoded.append(self.encoders[m](inputs[m]))

        fused = self.fusion(encoded)
        hier_embeddings = self.hierarchy(fused)
        out = self.classifier(hier_embeddings[-1])  # Use last-level embedding
        return out, hier_embeddings
