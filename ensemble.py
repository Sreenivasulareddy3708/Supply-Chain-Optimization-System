# ensemble.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import PatchTST, ResidualXGBoost
from features import create_features
from config import PATCHTST_CONFIG

class ErrorAdaptiveEnsemble:
    def __init__(self, seq_len=365):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_model = PatchTST(config=PATCHTST_CONFIG, seq_len=seq_len).to(self.device)
        self.xgb_model = ResidualXGBoost()
        
        # Adaptive Weights
        self.w_patch = 0.5
        self.w_xgb = 0.5
        
    def _inverse_weight(self, error_1, error_2):
        """Helper for Eq in Section III.E"""
        w1 = (1 / error_1) / (1/error_1 + 1/error_2)
        return w1

    def train_ensemble(self, train_loader, val_data_full, val_loader):
        """
        Implementation of Algorithm 1: Error-Adaptive Ensemble Training
        """
        print(f"Training PatchTST on {self.device}...")
        
        # --- Step 2: Train PatchTST ---
        optimizer = torch.optim.Adam(self.patch_model.parameters(), lr=PATCHTST_CONFIG['learning_rate'])
        criterion = nn.MSELoss()
        
        self.patch_model.train()
        for epoch in range(PATCHTST_CONFIG['epochs']):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device).unsqueeze(-1)
                batch_y = batch_y.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.patch_model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # (Optional) Print epoch loss or implement Early Stopping here
        
        # --- Step 2c: Generate Validation Forecast (Trend) ---
        self.patch_model.eval()
        val_x_tensor = torch.tensor(val_data_full['x']).float().to(self.device).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            y_val_patch = self.patch_model(val_x_tensor).cpu().numpy().flatten()
            
        y_val_actual = val_data_full['y']

        # --- Step 3: Compute Residuals ---
        # Residuals = Actual - PatchTST_Trend
        # Note: We need residuals for TRAINING XGBoost (from train set) 
        # and VALIDATION (to calculate weights)
        
        # For simplicity, assuming we have extracted train residuals:
        # train_residuals = y_train_actual - train_patch_preds
        # Here we focus on the validation/weighting logic logic provided in snippet
        
        val_residuals = y_val_actual - y_val_patch
        
        # --- Step 5: Train XGBoost on Residuals ---
        # Note: You need to generate features for Training data and Validation data
        # taking the raw df and adding the 'patchtst_pred' column.
        
        # Mocking feature generation call for clarity
        # X_train_feats = create_features(train_df, train_patch_preds)
        # X_val_feats = create_features(val_df, y_val_patch)
        
        print("Training XGBoost on residuals...")
        # self.xgb_model.fit(X_train_feats, train_residuals, X_val=X_val_feats, y_val=val_residuals)
        
        # --- Step 5c: Generate Validation Residual Forecast ---
        # y_val_xgb_resid = self.xgb_model.predict(X_val_feats)
        
        # Placeholder for actual xgb output for the logic flow
        y_val_xgb_resid = np.zeros_like(val_residuals) # Replace with actual prediction
        
        # --- Step 6: Calculate Adaptive Weights (Section III.E) ---
        
        # Calculate Errors for PatchTST (Raw)
        rmse_p = np.sqrt(mean_squared_error(y_val_actual, y_val_patch))
        mae_p = mean_absolute_error(y_val_actual, y_val_patch)
        
        # Calculate Errors for XGBoost (Residual Prediction)
        # Note: The paper compares RMSE of the model's contribution. 
        # Strictly, we compare the error of PatchTST vs Error of (PatchTST+XGB) or
        # as per paper: RMSE of PatchTST vs RMSE of XGBoost Residuals.
        # Paper citation : RMSE_XGBoost calculated on (r - r_hat)
        
        rmse_x = np.sqrt(mean_squared_error(val_residuals, y_val_xgb_resid))
        mae_x = mean_absolute_error(val_residuals, y_val_xgb_resid)
        
        # Inverse Error Weighting
        w_p_1 = self._inverse_weight(rmse_p, rmse_x)
        w_p_2 = self._inverse_weight(mae_p, mae_x)
        
        # Average
        self.w_patch = (w_p_1 + w_p_2) / 2
        self.w_xgb = 1 - self.w_patch
        
        print(f"Adaptive Weights Calculated: PatchTST={self.w_patch:.4f}, XGBoost={self.w_xgb:.4f}")

    def predict(self, x_input, feature_df_input):
        """
        Final Inference Step
        """
        # 1. Trend Forecast
        self.patch_model.eval()
        with torch.no_grad():
            trend_pred = self.patch_model(x_input).cpu().numpy().flatten()
            
        # 2. Residual Forecast
        # Update feature matrix with the new trend prediction
        feature_df_input['patchtst_pred'] = trend_pred
        # Generate other features (lags etc) here...
        
        resid_pred = self.xgb_model.predict(feature_df_input)
        
        # 3. Combine
        final_pred = trend_pred + (self.w_xgb * resid_pred)
        
        # 4. Non-negativity constraint
        final_pred = np.maximum(0, final_pred)
        
        return final_pred
