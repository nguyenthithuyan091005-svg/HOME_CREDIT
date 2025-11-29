# src/generate_llm_explanations.py
import os
import pandas as pd
from groq import Groq  # pip install groq

from gen_prompt import build_feature_glossary, build_explanation_prompt

# ==========================
# 0. Config / kiểm tra môi trường
# ==========================
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY is None:
    raise ValueError("❌ Chưa có GROQ_API_KEY trong biến môi trường. Hãy set bằng: $env:GROQ_API_KEY=\"...\"")

client = Groq(api_key=API_KEY)

# ==========================
# 1. Load dữ liệu cần thiết
# ==========================
TOP5_PATH = "data/top5_per_customer_100.csv"
PRED_PATH = "data/predictions_output_100.csv"
RISK_PATH = "data/risk_scores_100.csv"
TOP30_PATH = "data/top30_features_100.csv"
OUT_DIR = "data/explanations_100"

for p in (TOP5_PATH, PRED_PATH, RISK_PATH, TOP30_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Không tìm thấy file cần thiết: {p}. Hãy kiểm tra lại đường dẫn: {p}")

top5_df = pd.read_csv(TOP5_PATH)
predictions_df = pd.read_csv(PRED_PATH)
risk_df = pd.read_csv(RISK_PATH)

# glossary cho 30 feature
glossary = build_feature_glossary(top30_path=TOP30_PATH)

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# 2. Hàm trợ giúp: tạo kết luận tạm thời theo score
# ==========================
def make_default_conclusion_line(score):
    """
    Trả về câu kết luận tạm thời tùy theo mức rủi ro (score).
    Các ngưỡng chỉ mang tính minh họa trong phạm vi bài toán học thuật.
    """
    if score is None:
        return (
            "Kết luận tạm thời: Không đủ thông tin về xác suất rủi ro. "
            "Không nên phê duyệt tự động, cần thẩm định bổ sung hồ sơ và phỏng vấn khách hàng."
        )

    if score < 0.3:
        return (
            "Kết luận tạm thời: Khách hàng thuộc nhóm rủi ro thấp theo ngưỡng minh họa. "
            "Có thể xem xét phê duyệt, nhưng vẫn cần tuân thủ đầy đủ quy trình kiểm tra hồ sơ "
            "và xác minh thông tin theo quy định nội bộ."
        )
    elif 0.3 <= score <= 0.5:
        return (
            "Kết luận tạm thời: Khách hàng thuộc nhóm rủi ro trung bình theo ngưỡng minh họa. "
            "Không nên phê duyệt tự động. Cần thẩm định bổ sung hồ sơ, kiểm tra kỹ thu nhập, "
            "nghĩa vụ trả nợ hiện tại và lịch sử vay trước khi ra quyết định."
        )
    else:  # score > 0.5
        return (
            "Kết luận tạm thời: Khách hàng thuộc nhóm rủi ro cao theo ngưỡng minh họa. "
            "Không nên phê duyệt tự động. Cần thẩm định bổ sung rất kỹ, có thể xem xét giảm hạn mức vay, "
            "yêu cầu thêm tài sản bảo đảm hoặc các điều kiện ràng buộc khác nếu vẫn muốn hỗ trợ khách hàng."
        )

# ==========================
# 3. Loop qua từng khách hàng
# ==========================
for idx, row in top5_df.iterrows():
    sk_id = row["SK_ID_CURR"]
    top5_feats = row[1:].tolist()  # các cột top1..top5

    # --- lấy risk_score cho khách này ---
    risk_row = risk_df[risk_df["SK_ID_CURR"] == sk_id]
    score = None
    if not risk_row.empty:
        try:
            score = float(risk_row["risk_score"].values[0])
        except Exception:
            score = None

    # --- lấy giá trị feature của 5 biến này ---
    pred_row = predictions_df[predictions_df["SK_ID_CURR"] == sk_id]
    if pred_row.empty:
        print(f"⚠️ Không tìm thấy SK_ID_CURR = {sk_id} trong {PRED_PATH}, bỏ qua.")
        continue

    feature_values = pred_row[top5_feats].to_dict("records")[0]

    # --- build prompt XAI chuẩn nghiệp vụ ---
    base_prompt = build_explanation_prompt(
        sk_id,
        top5_feats,
        glossary,
        feature_values=feature_values,
        score=score,
    )

    full_prompt = (
        base_prompt
        + "\n\nDữ liệu cụ thể của khách hàng cho 5 biến quan trọng (tên_feature: giá trị_thực_tế):\n"
        + str(feature_values)
        + "\n\nHãy sử dụng đúng các giá trị trên, bám sát cấu trúc 5 phần đã mô tả (1️⃣ đến 5️⃣). "
          "Ưu tiên rõ ràng, mạch lạc, dễ hiểu cho nhân viên tín dụng. Nếu thiếu thông tin ở phần nào, "
          "vẫn cần ghi tiêu đề phần đó và ghi 'Không đủ dữ liệu để phân tích chi tiết'."
    )

    # ==========================
    # 4. Gọi Groq LLM
    # ==========================
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên viên phân tích rủi ro tín dụng cao cấp. "
                        "Trình bày rõ ràng, mạch lạc, dễ hiểu cho nhân viên tín dụng. "
                        "Luôn giữ đủ 5 phần: Tổng quan rủi ro, Top 5 yếu tố, Nhận xét tổng quan, "
                        "Ví dụ minh họa, Kết luận & Khuyến nghị."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=800,
            temperature=0.25,
        )

        explain_text = response.choices[0].message.content.strip()

    except Exception as e:
        explain_text = f"❌ Lỗi khi gọi Groq API cho khách hàng {sk_id}: {e}"
        print(explain_text)

    # ==========================
    # 5. Hậu kiểm: đảm bảo luôn có dòng "Kết luận tạm thời"
    # ==========================
    if "Kết luận tạm thời" not in explain_text:
        conclusion_line = make_default_conclusion_line(score)
        explain_text = explain_text.rstrip() + "\n\n" + conclusion_line

    # ==========================
    # 6. Lưu file riêng cho từng khách hàng
    # ==========================
    out_path = os.path.join(OUT_DIR, f"{sk_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(explain_text)

    print(f"✅ Đã tạo file giải thích cho khách hàng {sk_id}: {out_path}")

print("🎉 Hoàn tất sinh file giải thích trong folder", OUT_DIR)
