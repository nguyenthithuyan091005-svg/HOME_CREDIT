import pandas as pd

shap_df = pd.read_csv("data/shap_values_100.csv")
top30 = pd.read_csv("data/top30_features_100.csv")["feature"].tolist()

# giữ SK_ID_CURR + top30 feature
shap_top30 = shap_df[["SK_ID_CURR"] + top30]

top5_list = []
for idx, row in shap_top30.iterrows():
    top5_feats = row[top30].abs().sort_values(ascending=False).head(5).index.tolist()
    top5_list.append([row["SK_ID_CURR"]] + top5_feats)

top5_df = pd.DataFrame(top5_list, columns=["SK_ID_CURR"] + [f"top{i+1}" for i in range(5)])
top5_df.to_csv("data/top5_per_customer_100.csv", index=False)
print("✅ Top5 feature mỗi khách hàng đã lưu vào top5_per_customer_100.csv")
