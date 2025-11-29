# Dự án AI: Dự đoán rủi ro vỡ nợ tín dụng để tăng cường an toàn phục vụ cho quy trình xét duyệt vay tự động

## 👥 Thành viên nhóm  
**Nhóm 10**

- Trịnh Thị Lan Anh (31231025048)  
- Nguyễn Thị Thúy An (31231024410)  
- Nguyễn Thị Mỹ Thảo (31231025046)  
- Phan Thụy Kiến An (31231021959)  
- Nguyễn Thị Huỳnh Như (31231570390)  
- Trần Thị Thanh Hoa (31231026119)  
- Trương Ngọc Như Ý (31231025497)

---
## 📁 Cấu trúc thư mục của dự án
HOME_CREDIT/
|
+-- src/
|   +-- utils.py                      # Công cụ hỗ trợ
|   +-- train_gb_model.py             # Huấn luyện Gradient Boosting
|   +-- predict_gb_model.py           # Sinh dự báo
|   +-- shap_analysis.py              # Phân tích SHAP
|   +-- select_top30.py               # Lọc Top 30 features
|   +-- select_top5_from_shap.py      # Lọc Top 5 theo khách hàng
|   +-- compute_risk_score.py         # Tính Credit Score
|   +-- gen_prompt.py                 # Tạo câu lệnh AI
|   +-- generate_llm_explanations.py  # Sinh báo cáo bằng LLM
|
+-- data/
|   +-- feature_cols.joblib               # Danh sách biến
|   +-- feature_importance_output.csv     # Feature importance
|   +-- gb_model.joblib                   # Model đã huấn luyện
|   +-- predictions_output_100.csv        # Xác suất vỡ nợ
|   +-- shap_values_100.csv               # SHAP values
|   +-- risk_scores_100.csv               # Credit Score
|   +-- top30_features_100.csv            # Top 30 features
|   +-- top5_per_customer_100.csv         # Top 5 theo khách hàng
|
+-- EDA.ipynb                    # EDA
+-- feature-engineering.ipynb    # Preprocessing
+-- fairness.ipynb               # Fairness checking
+-- latest_aiproject.ows         # Orange project
+-- explain_100/                 # Giải thích cho 100 khách hàng
|
+-- README.md
+-- .gitignore

---
## 🌐 URL trang web nhóm  
**Link Notion:**  
https://aiproject-nhom10.notion.site/AI-Project-Nh-m-10-28fc815c90a88150adddfe83c9250bd2  

---

## 🗂️ URL repository của dự án  
**GitHub:**  
https://github.com/nguyenthithuyan091005-svg/HOME_CREDIT  

---
