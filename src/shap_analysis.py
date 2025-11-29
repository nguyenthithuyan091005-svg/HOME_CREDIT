import pandas as pd
import shap
import joblib

# ------------------------
# 1. Load dữ liệu dự đoán
# ------------------------
df_pred = pd.read_csv("data/predictions_output_100.csv")

# ------------------------
# 2. Chọn feature
# ------------------------
drop_cols = ["SK_ID_CURR", "prediction", "proba"]
feature_cols = [c for c in df_pred.columns if c not in drop_cols]
X_pred = df_pred[feature_cols]

# ------------------------
# 3. Fix NaN và kiểu dữ liệu
# ------------------------
X_pred = X_pred.fillna(0)      # điền NaN = 0
X_pred = X_pred.astype(float)  # convert sang float

# ------------------------
# 4. Load model
# ------------------------
gb_model = joblib.load("data/gb_model.joblib")

# ------------------------
# 5. Tạo SHAP Explainer
# ------------------------
explainer = shap.Explainer(gb_model, X_pred)
shap_values = explainer(X_pred)

# ------------------------
# 6. Lưu SHAP values
# ------------------------
shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
shap_df["SK_ID_CURR"] = df_pred["SK_ID_CURR"]
shap_df.to_csv("data/shap_values_100.csv", index=False)

print("💾 SHAP values đã lưu vào data/shap_values_100.csv")
