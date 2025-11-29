# ==========================
# gen_prompt.py – FINAL XAI VERSION (TOP 30 FEATURES)
# ==========================

import pandas as pd
from utils import load_csv

# ----------------------------------------------------------
# 0) Từ điển tên tiếng Việt cho 30 features
# ----------------------------------------------------------

FEATURE_VI_DICT = {
    "BUREAU_STATUS_C_SUM_SUM": "Tổng số lần nợ xấu (trạng thái C - rủi ro cao) trong lịch sử tín dụng",
    "BUREAU_STATUS_C_MAX_SUM": "Giá trị nợ xấu lớn nhất (trạng thái C - rủi ro cao) trong các khoản vay",
    "DAYS_BIRTH": "Tuổi khách hàng (tính từ ngày sinh)",
    "PREV_AMT_CREDIT_MIN": "Khoản vay nhỏ nhất trong các hợp đồng vay trước đây",
    "AMT_ANNUITY": "Số tiền khách hàng phải trả góp định kỳ cho khoản vay hiện tại",
    "BUREAU_STATUS_C_SUM_MEAN": "Trung bình số lần nợ xấu (trạng thái C - rủi ro cao) trên các khoản vay",
    "BUREAU_CREDIT_ACTIVE_ACTIVE_MEAN": "Tỷ lệ các khoản vay đang còn hoạt động trong lịch sử tín dụng",
    "EXT_SOURCE_3": "Chỉ số tổng hợp từ nguồn dữ liệu bên ngoài (đánh giá rủi ro tổng hợp)",
    "OWN_CAR_AGE": "Tuổi của xe ô tô thuộc sở hữu khách hàng",
    "AMT_INCOME_TOTAL": "Tổng thu nhập của khách hàng (thường tính theo năm hoặc tháng tuỳ hệ thống)",
    "CODE_GENDER_F": "Giới tính nữ (biến nhị phân: 1 = nữ, 0 = nam)",
    "DAYS_EMPLOYED": "Số ngày khách hàng đã làm việc (thâm niên công việc)",
    "BUREAU_STATUS_C_MEAN_SUM": "Tổng trung bình số lần nợ xấu (trạng thái C - rủi ro cao) trên toàn bộ lịch sử",
    "BUREAU_STATUS_X_MEAN_SUM": "Tổng trung bình trạng thái X (không đủ dữ liệu / chưa báo cáo) trong lịch sử tín dụng",
    "DAYS_ID_PUBLISH": "Số ngày kể từ khi giấy tờ tùy thân được cấp/cập nhật",
    "PREV_AMT_APPLICATION_MIN": "Giá trị nhỏ nhất khách hàng từng xin vay trong quá khứ",
    "ENTRANCES_AVG": "Trung bình số lối ra/vào của tòa nhà/khu vực khách hàng sinh sống",
    "FLOORSMAX_AVG": "Số tầng cao nhất trung bình của tòa nhà/khu vực sinh sống",
    "AMT_GOODS_PRICE": "Giá trị hàng hóa/tài sản khách hàng mua bằng khoản vay",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Số người quen trong vòng xã hội có nợ quá hạn dưới 30 ngày",
    "AMT_CREDIT": "Số tiền khoản vay/tín dụng được cấp cho khách hàng",
    "LIVINGAREA_MODE": "Diện tích sinh hoạt (chuẩn hóa) nơi khách hàng sinh sống",
    "APARTMENTS_MEDI": "Chỉ số trung vị về chất lượng/cấp căn hộ nơi khách hàng ở",
    "REGION_POPULATION_RELATIVE": "Mật độ dân số tương đối của khu vực khách hàng sinh sống",
    "CC_AMT_BALANCE_STD": "Độ biến động (độ lệch chuẩn) số dư trên thẻ tín dụng",
    "BUREAU_STATUS_X_MAX_SUM": "Giá trị lớn nhất của trạng thái X (không báo cáo) trong lịch sử tín dụng",
    "FLOORSMIN_AVG": "Số tầng thấp nhất trung bình của tòa nhà/khu vực sinh sống",
    "BUREAU_STATUS_1_SUM_SUM": "Tổng số lần trễ hạn 1 kỳ trong lịch sử tín dụng",
    "CC_CNT_INSTALMENT_MATURE_CUM_MAX": "Tổng số kỳ trả góp của thẻ tín dụng đến hạn (giá trị lớn nhất)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Số lần tra cứu thông tin tín dụng khách hàng trong 1 năm",
}

def get_feature_vi_name(feat_name: str) -> str:
    """Trả về tên tiếng Việt nếu có, ngược lại trả về tên gốc."""
    return FEATURE_VI_DICT.get(feat_name, feat_name)

# ----------------------------------------------------------
# 1) Mô tả gợi ý/nghiệp vụ cho từng feature
# ----------------------------------------------------------

IMPACT_HINT_DICT = {
    "BUREAU_STATUS_C_SUM_SUM": (
        "Biến này đo tổng số lần khách hàng bị ghi nhận ở trạng thái nợ xấu (C - rủi ro cao) "
        "trong toàn bộ lịch sử tín dụng. Giá trị càng cao cho thấy lịch sử nợ xấu nhiều, "
        "làm tăng rủi ro không trả nợ."
    ),
    "BUREAU_STATUS_C_MAX_SUM": (
        "Đại diện cho lần nợ xấu nghiêm trọng nhất. Giá trị cao cho thấy từng có giai đoạn "
        "khách hàng chậm trả hoặc không trả nợ đáng kể, làm tăng rủi ro."
    ),
    "DAYS_BIRTH": (
        "Số ngày tính từ ngày sinh. Trị tuyệt đối càng lớn thì tuổi càng cao. "
        "Độ tuổi có thể liên quan đến sự ổn định thu nhập và hành vi tín dụng."
    ),
    "PREV_AMT_CREDIT_MIN": (
        "Khoản tín dụng nhỏ nhất từng được cấp trong quá khứ. Có thể phản ánh mức độ "
        "thận trọng hoặc quy mô vay trước đây của khách hàng."
    ),
    "AMT_ANNUITY": (
        "Số tiền khách hàng phải trả góp theo kỳ cho khoản vay hiện tại. "
        "Nếu con số này lớn so với thu nhập thực tế, áp lực trả nợ sẽ cao → tăng rủi ro."
    ),
    "BUREAU_STATUS_C_SUM_MEAN": (
        "Số lần nợ xấu (C) trung bình trên mỗi khoản vay. Giá trị cao cho thấy không chỉ một "
        "mà nhiều khoản vay của khách hàng có vấn đề → tăng rủi ro."
    ),
    "BUREAU_CREDIT_ACTIVE_ACTIVE_MEAN": (
        "Tỷ lệ khoản vay hiện còn đang hoạt động. Tỷ lệ quá cao có thể cho thấy khách hàng đang gánh nhiều khoản vay cùng lúc."
    ),
    "EXT_SOURCE_3": (
        "Chỉ số tổng hợp từ nguồn dữ liệu bên ngoài (ví dụ: chấm điểm tín dụng bên thứ ba). "
        "Giá trị càng cao thường gắn với mức độ rủi ro thấp hơn."
    ),
    "OWN_CAR_AGE": (
        "Tuổi của xe ô tô thuộc sở hữu khách hàng. Xe đã sử dụng lâu năm thường có giá trị tài sản thấp hơn."
    ),
    "AMT_INCOME_TOTAL": (
        "Tổng thu nhập của khách hàng. Thu nhập ổn định, tương xứng với nghĩa vụ trả nợ giúp giảm rủi ro."
    ),
    "CODE_GENDER_F": (
        "Biến giới tính (1 = nữ). Không phải lúc nào cũng mang tính quyết định, nhưng có thể liên quan đến một số mô hình hành vi trả nợ."
    ),
    "DAYS_EMPLOYED": (
        "Số ngày đã làm việc tại nơi hiện tại. Thâm niên cao cho thấy sự ổn định về công việc và thu nhập."
    ),
    "BUREAU_STATUS_C_MEAN_SUM": (
        "Tổng trung bình trạng thái nợ xấu C trên toàn bộ lịch sử, kết hợp cả số lượng và phân bố theo thời gian."
    ),
    "BUREAU_STATUS_X_MEAN_SUM": (
        "Mức độ xuất hiện trạng thái X (không báo cáo/không đủ dữ liệu). Nếu quá nhiều, "
        "có thể gây khó khăn cho đánh giá lịch sử tín dụng."
    ),
    "DAYS_ID_PUBLISH": (
        "Thời gian kể từ khi giấy tờ tùy thân được cấp/cập nhật. Giấy tờ mới có thể liên quan đến thay đổi thông tin, "
        "cần kiểm tra thêm."
    ),
    "PREV_AMT_APPLICATION_MIN": (
        "Số tiền xin vay nhỏ nhất trong các lần nộp hồ sơ trước đây. Có thể phản ánh "
        "xu hướng vay khoản nhỏ, ít rủi ro hoặc mới bắt đầu lịch sử vay."
    ),
    "ENTRANCES_AVG": (
        "Trung bình Tần suất Di chuyển/Hoạt động của Khách hàng trong khu vực sinh sống, gián tiếp phản ánh chất lượng và vị trí chỗ ở."
    ),
    "FLOORSMAX_AVG": (
        "Số tầng cao nhất trung bình. Có thể liên quan đến loại hình nhà ở và mức độ phát triển của khu vực, gián tiếp phản ánh giá trị tài sản, mức sống cũng như thu nhập của khách hàng."
    ),
    "AMT_GOODS_PRICE": (
        "Giá trị tài sản/hàng hóa được tài trợ bởi khoản vay. Giá trị càng lớn → rủi ro càng cao "
        "nếu thu nhập không tương xứng."
    ),
    "OBS_30_CNT_SOCIAL_CIRCLE": (
        "Số người trong vòng quan hệ xã hội có nợ quá hạn dưới 30 ngày. Môi trường xung quanh nhiều người nợ xấu "
        "có thể là dấu hiệu rủi ro."
    ),
    "AMT_CREDIT": (
        "Tổng số tiền khoản vay được duyệt. Khoản vay càng lớn, nếu không phù hợp với thu nhập → tăng rủi ro."
    ),
    "LIVINGAREA_MODE": (
        "Diện tích sinh hoạt (đã chuẩn hóa). Diện tích nhỏ trong khu vực chi phí sinh hoạt cao có thể tạo áp lực tài chính."
    ),
    "APARTMENTS_MEDI": (
        "Chỉ số trung vị về mức độ/loại hình căn hộ. Có thể phản ánh điều kiện sống và tài sản của khách hàng."
    ),
    "REGION_POPULATION_RELATIVE": (
        "Mật độ dân số tương đối khu vực sinh sống. Khu vực quá đông hoặc quá thưa thường mang những đặc điểm rủi ro khác nhau."
    ),
    "CC_AMT_BALANCE_STD": (
        "Độ dao động số dư thẻ tín dụng. Biến động lớn có thể phản ánh hành vi chi tiêu không ổn định."
    ),
    "BUREAU_STATUS_X_MAX_SUM": (
        "Giá trị lớn nhất của trạng thái X (không báo cáo). Nếu tập trung nhiều trong một khoản vay, "
        "có thể là khoảng thời gian khó đánh giá."
    ),
    "FLOORSMIN_AVG": (
        "Tầng thấp nhất trung bình nơi khách hàng sinh sống. Một số khu vực tầng thấp có thể rủi ro hơn (ẩm thấp, ngập, an ninh...), "
        "gián tiếp phản ánh chất lượng khu vực cư trú, cũng phản ánh mức sống, mức thu nhập của khách hàng."
    ),
    "BUREAU_STATUS_1_SUM_SUM": (
        "Tổng số lần khách hàng trễ hạn 1 kỳ. Đây là tín hiệu cảnh báo sớm về thói quen trả nợ không đúng hạn."
    ),
    "CC_CNT_INSTALMENT_MATURE_CUM_MAX": (
        "Tổng số kỳ trả góp thẻ tín dụng tới hạn (giá trị lớn nhất). Nhiều kỳ tới hạn cùng lúc tạo áp lực trả nợ."
    ),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (
        "Số lần tổ chức tín dụng tra cứu thông tin khách hàng trong 1 năm. Việc bị hỏi nhiều lần "
        "có thể cho thấy khách hàng đang vay nhiều nơi hoặc có nhu cầu vay thêm liên tục."
    ),
}

def get_feature_hint(feat_name: str) -> str:
    return IMPACT_HINT_DICT.get(
        feat_name,
        "Biến này ảnh hưởng đến khả năng trả nợ (thu nhập, lịch sử nợ, hành vi thanh toán hoặc tài sản)."
    )

# ----------------------------------------------------------
# 2) Xây dựng “từ điển giải thích” cho top 30 features
# ----------------------------------------------------------

# ----------------------------------------------------------
# 2) Tạo Glossary tự động cho 30 features (LLM sẽ dùng)
# ----------------------------------------------------------

def build_feature_glossary(top30_path="data/top30_features_100.csv"):
    """
    Hàm này đọc danh sách 30 feature và trả về mô tả Tiếng Việt + giải thích nghiệp vụ.
    Ưu tiên lấy mô tả từ IMPACT_HINT_DICT.
    Nếu feature không nằm trong dictionary → tạo mô tả mặc định.
    """
    df = load_csv(top30_path)
    glossary = {}

    for f in df["feature"]:
        vi_name = get_feature_vi_name(f)          # tên tiếng Việt
        impact = get_feature_hint(f)              # mô tả nghiệp vụ

        glossary[f] = f"{vi_name}. {impact}"

    return glossary

def build_explanation_prompt(sk_id, top5_features, glossary, feature_values=None, score=None):
    """
    Sinh prompt giải thích XAI cho một khách hàng cụ thể.
    - sk_id: mã khách hàng
    - top5_features: list tên 5 feature quan trọng nhất
    - glossary: dict {feature: mô tả tiếng Việt + nghiệp vụ}
    - feature_values: dict {feature: value} (giá trị thực tế của 5 feature)
    - score: risk_score (xác suất rủi ro, 0–1)
    """

    # ===== 1) Chuẩn bị Risk Score, % và nhóm rủi ro =====
    if score is None:
        score_str = "N/A"
        score_pct_str = "N/A"
        risk_group = "không xác định"
        risk_action = "Cần thực hiện quy trình thẩm định theo quy định chung."
    else:
        score_str = f"{score:.3f}"

        # chuyển sang % và format kiểu Việt Nam: 75,5%
        score_pct = round(score * 100, 1)          # 0.755 -> 75.5
        score_pct_str = str(score_pct).replace(".", ",") + "%"

        # gán nhóm rủi ro theo ngưỡng minh họa
        if score > 0.5:
            risk_group = "rủi ro cao"
            risk_action = (
                "Cần được thẩm định bổ sung rất kỹ (thu nhập, lịch sử vay, tài sản bảo đảm) "
                "trước khi ra quyết định phê duyệt khoản vay."
            )
        elif score >= 0.3:
            risk_group = "rủi ro trung bình"
            risk_action = (
                "Cần xem thêm giấy tờ chứng minh thu nhập, sao kê tài khoản, lịch sử vay và các thông tin bổ sung "
                "trước khi ra quyết định."
            )
        else:
            risk_group = "rủi ro thấp"
            risk_action = (
                "Có thể xem xét phê duyệt nếu các điều kiện và giấy tờ khác đạt yêu cầu, "
                "nhưng vẫn phải tuân thủ đầy đủ quy trình thẩm định."
            )

    # ===== 2) Xây prompt =====
    text = f"""
Giải thích hồ sơ khách hàng {sk_id}

1️⃣ Tổng quan rủi ro (Risk Score)

- Risk Score (xác suất rủi ro): {score_str}

Giải thích:
- Đây là xác suất mô hình dự đoán khách hàng không trả được nợ đúng hạn.
- Giá trị càng gần 1 thì rủi ro càng cao, càng gần 0 thì rủi ro càng thấp.
- Các ngưỡng dưới đây chỉ mang tính minh họa trong phạm vi bài toán học thuật (không phải quy định chính thức của tổ chức tín dụng):
  + > 0.5  → Nhóm rủi ro cao → cần thẩm định bổ sung rất kỹ trước khi duyệt.
  + 0.3 – 0.5 → Nhóm rủi ro trung bình → cần xem thêm giấy tờ, kiểm tra thu nhập, lịch sử vay.
  + < 0.3 → Nhóm rủi ro thấp → khách hàng có khả năng trả nợ tốt hơn, nhưng vẫn cần quy trình kiểm tra rõ ràng.
"""

    # chèn câu Đánh giá tự động nếu có score
    if score is not None:
        text += f"""
Đánh giá: Khách hàng này có {score_pct_str} khả năng rủi ro không trả nợ, thuộc nhóm {risk_group} theo các ngưỡng minh họa trên, {risk_action}
"""

    text += f"""

2️⃣ Top 5 yếu tố ảnh hưởng mạnh nhất đến quyết định cho vay hồ sơ khách hàng {sk_id}

Dưới đây là 5 biến quan trọng nhất (theo SHAP) mà mô hình dùng để ra quyết định cho khách hàng này, kèm giá trị cụ thể:
"""

    # 3) Liệt kê từng feature + tên tiếng Việt + giá trị + mô tả
    for f in top5_features:
        vi_name = get_feature_vi_name(f)
        desc = glossary.get(f, get_feature_hint(f))
        val_str = "N/A"
        if feature_values is not None and f in feature_values:
            val_str = feature_values[f]

        text += f"""
- **{f}** – {vi_name}
  - Giá trị hiện tại: {val_str}
  - Ý nghĩa nghiệp vụ: {desc}
  - Hãy giải thích biến này với giá trị cụ thể trên đang có xu hướng làm **tăng rủi ro** hay **giảm rủi ro** đối với hồ sơ khách hàng này, bằng ngôn ngữ đơn giản, dễ hiểu cho nhân viên tín dụng.
"""

    text += f"""

3️⃣ Nhận xét tổng quan tác động

Hãy tóm tắt lại tác động của 5 biến trên:
- Nêu rõ biến/nhóm biến nào là nguồn rủi ro chính (ví dụ: lịch sử nợ xấu, khoản vay lớn, áp lực trả góp, tra cứu tín dụng nhiều lần, v.v.).
- Nêu rõ biến/nhóm biến nào có tác dụng hỗ trợ (ví dụ: thu nhập ổn định, ít nợ xấu, tài sản rõ ràng, môi trường sống ổn định...).
- Diễn đạt theo cách dễ hiểu cho nhân viên tín dụng, tránh quá kỹ thuật.

4️⃣ Ví dụ minh họa về khách hàng

Hãy mô tả lại hồ sơ khách hàng {sk_id} dưới dạng một đoạn mô tả ngắn (như kể về một trường hợp thực tế, không nêu tên thật):
- Khách hàng này có lịch sử tín dụng như thế nào (ổn định hay nhiều lần nợ xấu)?
- Nghĩa vụ trả nợ hiện tại có tạo áp lực lớn lên thu nhập hay không?
- Tài sản, nơi ở, môi trường sống có giúp giảm bớt rủi ro hay không?
- Kết nối các ý trên với Risk Score để người đọc dễ hình dung.

5️⃣ Kết luận & Khuyến nghị cho nhân viên tín dụng

Hãy đưa ra:
- Đánh giá tổng quan: khách hàng thuộc nhóm rủi ro thấp / trung bình / cao (theo ngưỡng minh họa đã nêu).
- Khuyến nghị hành động:
  - Có nên thẩm định bổ sung không? Nếu có, cần bổ sung những thông tin/giấy tờ gì (bảng lương, sao kê tài khoản, lịch sử vay nơi khác, định giá tài sản, v.v.)?
  - Có nên cân nhắc giảm hạn mức vay, tăng tài sản bảo đảm, yêu cầu đồng bảo lãnh, hoặc điều chỉnh điều kiện vay (kỳ hạn, lãi suất...) hay không?
- Viết bằng tiếng Việt, giọng văn chuyên nghiệp, rõ ràng, súc tích, dễ hiểu cho nhân viên tín dụng.
"""

    return text
