# models.py
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
from config import PATCHTST_CONFIG, XGBOOST_PARAMS

class PatchTST(nn.Module):
    """
    Implementation of Component 1: PatchTST for Trend Modeling
    Based on Section III.C and Appendix A.1
    """
    def __init__(self, config=PATCHTST_CONFIG, seq_len=365): # Assuming 1 year lookback
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = config['forecast_horizon']
        self.patch_len = config['patch_len']
        self.stride = config['stride']
        self.d_model = config['d_model']
        
        # Calculate number of patches
        self.num_patches = int((seq_len - self.patch_len) / self.stride) + 1
        
        # 1. Patch Embedding & Linear Projection
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # 2. Positional Encoding (Learnable)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        self.dropout = nn.Dropout(config['dropout'])
        
        # 3. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=config['n_heads'], 
            dim_feedforward=config['d_ff'], 
            dropout=config['dropout'], 
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['e_layers'])
        
        # 4. Forecast Head
        # Projects flattened embeddings to forecast horizon
        self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, 1] (Univariate)
        
        # Patching
        # Extract patches: This is a simplified view. 
        # Real implementation often uses unfold.
        B, L, C = x.shape
        x = x.squeeze(-1) # [B, L]
        
        # Create patches manually for clarity (or use unfold)
        # Result shape: [B, Num_Patches, Patch_Len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # Embedding
        x_enc = self.patch_embedding(patches) # [B, Num_Patches, d_model]
        
        # Add Positional Encoding
        x_enc = x_enc + self.position_embedding
        x_enc = self.dropout(x_enc)
        
        # Transformer Encoder
        z = self.encoder(x_enc) # [B, Num_Patches, d_model]
        
        # Flatten and Project
        z = z.reshape(z.shape[0], -1) # [B, Num_Patches * d_model]
        forecast = self.head(z) # [B, Horizon]
        
        return forecast.unsqueeze(-1)

class ResidualXGBoost:
    """
    Wrapper for Component 2: XGBoost for Residual Spike Correction
    """
    def __init__(self):
        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS)

    def fit(self, X_features, y_residuals, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_features, 
            y_residuals, 
            eval_set=eval_set, 
            verbose=False
        )
        
    def predict(self, X_features):
        return self.model.predict(X_features)
