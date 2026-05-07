# config.py

# PatchTST Hyperparameters (Appendix B.1)
PATCHTST_CONFIG = {
    'patch_len': 16,
    'stride': 16,
    'd_model': 128,
    'n_heads': 4,
    'e_layers': 3,
    'd_ff': 256,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'forecast_horizon': 90  # H variable in paper
}

# XGBoost Hyperparameters (Appendix B.2)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'early_stopping_rounds': 20
}
