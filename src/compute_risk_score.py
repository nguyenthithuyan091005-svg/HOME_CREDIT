import pandas as pd
from utils import load_csv, save_csv

def compute_risk(pred_csv='data/predictions_output_100.csv', 
                 out_csv='data/risk_scores_100.csv', 
                 proba_col='proba'):
    df = load_csv(pred_csv)
    if proba_col not in df.columns:
        raise ValueError(f'Column {proba_col} not found in predictions file.')
    out = df[['SK_ID_CURR', proba_col]].copy()
    out.columns = ['SK_ID_CURR', 'risk_score']
    save_csv(out, out_csv)
    return out

if __name__=='__main__':
    compute_risk()
