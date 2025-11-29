import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

DATA_PATH = "data/train_clean_34k.csv"
df_train = pd.read_csv(DATA_PATH)

drop_cols = ["SK_ID_CURR", "TARGET"]
feature_cols = [c for c in df_train.columns if c not in drop_cols]

print(f"Kích thước tập huấn luyện: {df_train.shape}")
print(f"Các kiểu dữ liệu không phải số:\n{df_train[feature_cols].dtypes[df_train[feature_cols].dtypes != 'int64'][df_train[feature_cols].dtypes != 'float64']}")
X_train = df_train[feature_cols]
y_train = df_train["TARGET"]

print("⚡ Bắt đầu huấn luyện Gradient Boosting Model...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)

os.makedirs("data", exist_ok=True)
joblib.dump(gb_model, "data/gb_model.joblib")
joblib.dump(feature_cols, "data/feature_cols.joblib")  # 🔹 LƯU THÊM DÒNG NÀY
print("✅ Model & feature_cols đã lưu vào data/")
