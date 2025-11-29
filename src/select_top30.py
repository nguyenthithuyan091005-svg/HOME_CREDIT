import pandas as pd

shap_df = pd.read_csv("data/shap_values_100.csv")

# loại bỏ SK_ID_CURR
feature_cols = [c for c in shap_df.columns if c != "SK_ID_CURR"]

# tính trung bình |SHAP| của mỗi feature
top30_feats = shap_df[feature_cols].abs().mean().sort_values(ascending=False).head(30)
top30_feats = top30_feats.reset_index().rename(columns={"index": "feature", 0:"mean_shap"})
top30_feats.to_csv("data/top30_features_100.csv", index=False)
print("✅ Top30 feature chung đã lưu vào top30_features_100.csv")
