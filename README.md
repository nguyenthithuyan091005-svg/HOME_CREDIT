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
│
├── src/
│   ├── utils.py                     # Công cụ hỗ trợ
│   ├── train_gb_model.py            # Huấn luyện mô hình Gradient Boosting
│   ├── predict_gb_model.py          # Sinh dự báo
│   ├── shap_analysis.py             # Phân tích SHAP (giải thích mô hình)
│   ├── select_top30.py              # Lọc Top 30 features quan trọng nhất
│   ├── select_top5_from_shap.py     # Lọc Top 5 feature theo từng khách hàng
│   ├── compute_risk_score.py        # Tính Credit Score
│   ├── gen_prompt.py                # Tạo câu lệnh AI (Prompt)
│   ├── generate_llm_explanations.py # Sinh báo cáo giải thích tự động bằng LLM
│
├── data/
│   ├── feature_cols.joblib              # Danh sách biến đầu vào
│   ├── feature_importance_output.csv    # Mức độ quan trọng của biến
│   ├── gb_model.joblib                  # Mô hình Gradient Boosting đã huấn luyện
│   ├── predictions_output_100.csv       # Xác suất vỡ nợ của 100 khách hàng mẫu
│   ├── shap_values_100.csv              # SHAP values của 100 khách hàng
│   ├── risk_scores_100.csv              # Credit Score
│   ├── top30_features_100.csv           # 30 đặc trưng mạnh nhất
│   ├── top5_per_customer_100.csv        # 5 đặc trưng quan trọng nhất theo khách hàng
│
├── EDA.ipynb                        # Phân tích EDA
├── feature-engineering.ipynb        # Tiền xử lý & tạo features
├── fairness.ipynb                   # Kiểm tra Fairness mô hình
├── latest_aiproject.ows             # Mô hình xây dựng bằng Orange
├── explain_100/                     # Giải thích kết quả cho 100 khách hàng
│
├── README.md
└── .gitignore


---
## 🌐 URL trang web nhóm  
**Link Notion:**  
https://aiproject-nhom10.notion.site/AI-Project-Nh-m-10-28fc815c90a88150adddfe83c9250bd2  

---

## 🗂️ URL repository của dự án  
**GitHub:**  
https://github.com/nguyenthithuyan091005-svg/HOME_CREDIT  

---
