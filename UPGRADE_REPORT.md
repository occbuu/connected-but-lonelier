Bản ghi nâng cấp ban đầu; đã có bản sửa tiếp theo ở PEER_REVIEW_RESPONSE.md. Các kết luận SHAP liên chu kỳ và đường toán Việt Nam được hạ/loại bỏ theo kiểm tra OECD mới, danh mục hiện là 64 nguồn và PDF ẩn danh.

# Paper 1: nâng cấp bản thảo ngày 14/10/2026

Các phép tính chạy trên máy local, từ dữ liệu đã trích xuất. LightGBM/SHAP và thống kê khảo sát chạy CPU (4 luồng cho mô hình), không cần gọi cloud; môi trường GPU local đã dùng cho paper 2. Không tạo dữ liệu hay kết quả giả.

## Thay đổi khoa học quan trọng

1. Belonging: dùng B3 dương làm anchor, giải phóng intercept của B1/B4/B6 âm, giữ correlated residuals cho wording. Tất cả mô hình hội tụ. Partial scalar CFI=0,9162, ΔCFI so metric=−0,0600; vẫn không đạt tiêu chí −0,010. Ước tính 2022 so 2012: full scalar −0,2502 SD, partial −0,0348 SD; bỏ 2018: −0,0421 SD. Không báo cáo độ giảm tiềm ẩn lớn/vững nữa. CFA weighted-moment ML xem item ordinal như continuous và không dùng chi-square khảo sát.
2. DFL chạy năm đặc tả, refit cả 80 BRR của mỗi chu kỳ độc lập (160 perturbations), không dùng xấp xỉ 20 replicate cũ. Composition là gender/grade/repeat/books; access thêm computer/internet; missing covariate riêng, outcome available-case. Loneliness tăng 8,267 pp, SE 0,719; access contribution −1,236 pp, SE 1,765 ở untrimmed, âm ở cả năm cách. Stage-2 ESS chỉ 367/6068; 05/95 propensity clipping 637; 5/95 ratio winsorisation 1123. Cắt tỷ số là winsorisation, không phải loại quan sát hay thay sang overlap population. DFL chỉ phân rã yếu tố đo được, không loại trừ nhân quả.
3. 199 hoán vị joint-access trong gender×ESCS quintile: chẩn đoán estimator, không phải placebo outcome độc lập hay kiểm định causal null. Reading liên hệ ICTRES +17,76 điểm/SD (SE 2,36), mathematics +18,17 (SE 2,48); reading không thích hợp làm biến không liên quan công nghệ.
4. GroupKFold theo 178 trường, 6068 học sinh toán/5934 lo âu. Calibration regression cùng reliability; phân loại riêng dưới Level 2 (420,07 điểm) cho từng 10 PV. Brier trung bình 0,1637 vs fold-training prevalence baseline 0,2027; không tính Brier trên outcome continuous. Không biến SHAP thành hiệu ứng nhân quả. Không báo cáo CI chưa được tính.
5. COVID: cô đơn tăng trước dịch 2012–2018 6,55 pp, khoảng 2018–2022 1,72 pp. Không có dữ liệu PISA 2020/2021; không nhận diện hiệu ứng đóng trường. Curriculum 2018 lộ trình grade 10 2022/23 là bối cảnh, chưa chứng minh cohort PISA2022 thực sự học chương trình đó. Thi quốc gia thay từ 2015 cũng là cơ chế thay thế chưa ước lượng.

## Trình bày và tài liệu

- Các hình dùng PDF vector và PNG 300 dpi, conceptual framework tách `latex/figures/framework.tex`. Multi-panel có nhãn a/b/…; số mẫu, BRR/PV, missing và ký hiệu p/q bổ sung trong captions, bảng và phụ lục.
- APA 7 với biblatex-apa/Biber, journal+volume nghiêng và hanging 0,5 inch. Volume đậm không phải APA 7 nên không áp dụng yêu cầu đó dưới nhãn APA.
- Tài liệu DOI kiểm tra với Crossref, lưu JSON audit. 40 tài liệu đều được trích dẫn, bỏ Bernardo/Hu không dùng. Năm issue Twenge là 2026, không phải 2025. Không gọi OECD policy brief hoặc systematic review là meta-analysis PISA2022.
- Đã chỉnh tiếng Anh và tóm tắt Việt để đồng nhất kết luận/tense. Chưa có biên tập viên ngôn ngữ chuyên nghiệp rà soát; không tuyên bố đã có.
- Declared đúng OpenAI Codex hỗ trợ code, phân tích, ngôn ngữ và LaTeX; không gán ChatGPT-4 khi không có căn cứ.

## Chưa thể điền thay tác giả

Tác giả, đơn vị, email, CRediT và URL/DOI mã nguồn; chưa có URL theo xác nhận của người dùng. `publish_repository/` đã chuẩn bị để push, dữ liệu gốc dẫn nguồn thay vì redistribute. Tác giả chọn license, đọc và xác nhận kết quả/AI statement trước khi nộp. Không có tác động nhân quả được xác lập bởi các kiểm tra bổ sung.

Các CSV mới: `outputs/results_upgrade/U1...U6...`; `reference_audit.json` là metadata DOI, `reference_usage.json` là ledger sử dụng trích dẫn. Bản trước chỉnh được giữ trong `archive/pre_upgrade_2026-10-14/`.
