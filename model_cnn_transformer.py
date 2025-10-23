import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features):
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outputs = [
            fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]
        return outputs


class VietnamesePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(VietnamesePositionalEncoding, self).__init__()

        # Base positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add special encoding for Vietnamese tones and marks
        tone_encoding = torch.zeros(max_len, d_model)
        tone_encoding[:, : d_model // 4] = torch.sin(
            position * div_term[: d_model // 4] * 2
        )
        pe = pe + tone_encoding * 0.1

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # Learnable tone attention
        self.tone_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pos_enc = self.pe[:, : x.size(1)]
        tone_weights = self.tone_attention(pos_enc)
        return x + pos_enc * tone_weights


class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super(CNNEncoder, self).__init__()

        # Load pretrained ConvNeXt Large (độ chính xác cao nhất)
        self.backbone = timm.create_model(
            "convnext_large",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Get features from all 4 stages
        )

        # ConvNeXt-L feature channels: [192, 384, 768, 1536]
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[192, 384, 768, 1536], out_channels=d_model
        )

        # Final projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # Extract multi-scale features from ConvNeXt Large
        features = self.backbone(x)  # Returns list of 4 feature maps

        # Print shapes for debugging (comment out after confirming)
        # for i, feat in enumerate(features):
        #     print(f'Stage {i}:', feat.shape)

        # FPN processing
        fpn_features = self.fpn(features)

        # Concatenate and project
        B, C, H, W = fpn_features[0].shape
        combined = torch.cat(
            [
                F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
                for f in fpn_features
            ],
            dim=1,
        )

        feat = self.final_conv(combined)

        # Reshape to sequence format
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        return feat


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=36, dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = VietnamesePositionalEncoding(d_model, max_len)

        # Increased number of layers and heads
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Increased feedforward dimension
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # Changed to GELU activation
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection with layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)

        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        output = self.layer_norm(output)
        return self.output_layer(output)


class OCRModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super(OCRModel, self).__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,  # Increased number of attention heads
            num_layers=6,  # Increased number of layers
            dropout=0.1,
        )

        # Add label smoothing
        self.label_smoothing = 0.1

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz)) * float("-inf"), diagonal=1)
        return mask

    def forward(self, images, tgt_seq):
        memory = self.encoder(images)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq.size(1)).to(
            tgt_seq.device
        )
        output = self.decoder(tgt_seq, memory, tgt_mask=tgt_mask)
        return output

    def compute_loss(self, output, target):
        # Apply label smoothing
        n_classes = output.size(-1)
        one_hot = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.label_smoothing) + (
            self.label_smoothing / n_classes
        )
        log_prob = F.log_softmax(output, dim=-1)
        loss = (-smooth_one_hot * log_prob).sum(dim=-1).mean()
        return loss
