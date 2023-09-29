import torch
import torch.nn as nn

class FuzzyViT(nn.Module):
    def __init__(self, num_classes, embed_dim=768, num_heads=12, num_layers=12, dropout_prob=0.1, use_fuzzy_dropout=False):
        super(FuzzyViT, self).__init__()

        self.embed_dim = embed_dim  # For dynamic linear_patch_embed
        self.use_fuzzy_dropout = use_fuzzy_dropout
        self.dropout_prob = dropout_prob

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def position_masking(self, x, mode="random"):
        B, N, _ = x.size()
        if mode == "random":
            mask = torch.randint(0, 2, x.shape[:2]).bool().to(x.device)
            x = x.masked_fill(~mask.unsqueeze(-1), 0)

        elif mode == "cyclic":
            stride = N // 4  
            mask = torch.zeros(B, N).to(x.device)
            for i in range(0, N, stride):
                mask[:, i:i+stride//2] = 1  
            x = x.masked_fill(mask.bool().unsqueeze(-1), 0)

        elif mode == "fuzzy":
            mask = (torch.rand(x.shape[:2]) > 0.5).to(x.device)
            x = x.masked_fill(~mask.unsqueeze(-1), 0)

        return x

    def _process(self, x):
        B, C, T = x.size()

        # Calculate the patch embedding dimensions dynamically
        self.patch_size = T
        num_patches = T // self.patch_size
        
        if not hasattr(self, 'linear_patch_embed') or self.linear_patch_embed.in_features != C * self.patch_size:
            self.linear_patch_embed = nn.Linear(C * self.patch_size, self.embed_dim).to(x.device)
        if not hasattr(self, 'position_embed') or self.position_embed.size(0) != num_patches + 1:
            self.position_embed = nn.Parameter(torch.randn(num_patches + 1, self.embed_dim)).to(x.device)

        # Reshape to patches
        x = x.unfold(2, self.patch_size, self.patch_size).contiguous()
        x = x.view(B, num_patches, -1)
        x = self.linear_patch_embed(x)

        # Position Masking
        x = self.position_masking(x, mode="random")
        
        if self.use_fuzzy_dropout:
            dropout_mask = (torch.rand(self.position_embed.shape) > self.dropout_prob).type_as(self.position_embed)
            position_embed = self.position_embed * dropout_mask
        else:
            position_embed = self.position_embed

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + position_embed
        x = self.dropout(x)
        x = self.transformer(x)
        return x[:, 0]

    def forward(self, mel1, mel2, mel3):
        out1 = self._process(mel1)
        out1_expanded = out1.unsqueeze(1).expand(-1, mel2.size(2), -1).transpose(1, 2)
        concat1 = torch.cat([out1_expanded, mel2], dim=1)
        
        out2 = self._process(concat1)
        out2_expanded = out2.unsqueeze(1).expand(-1, mel3.size(2), -1).transpose(1, 2)
        concat2 = torch.cat([out2_expanded, mel3], dim=1)
        
        out3 = self._process(concat2)
        
        return self.mlp_head(out3)


