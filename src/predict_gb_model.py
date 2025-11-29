import pandas as pd
import joblib

# 1. Load model & feature list
gb_model = joblib.load("data/gb_model.joblib")
feature_cols = joblib.load("data/feature_cols.joblib")

# 2. Load data 48k rồi lấy 100 khách hàng đầu
df_pred = pd.read_csv("data/predictions_output.csv")
df_pred = df_pred.head(100)

# 3. Đảm bảo đủ hết các cột feature như lúc train
for col in feature_cols:
    if col not in df_pred.columns:
        print(f"⚠️ Thiếu cột {col} trong predictions_output.csv -> tạo tạm với 0")
        df_pred[col] = 0

# Nếu trong df_pred thừa cột nào ngoài feature_cols + SK_ID_CURR thì không sao
X_pred = df_pred[feature_cols]

# 4. Dự đoán
preds = gb_model.predict(X_pred)
probas = gb_model.predict_proba(X_pred)[:, 1]

df_pred["prediction"] = preds
df_pred["proba"] = probas

df_pred.to_csv("data/predictions_output_100.csv", index=False)
print("💾 Đã lưu predictions_output_100.csv")
