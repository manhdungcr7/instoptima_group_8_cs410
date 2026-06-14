# Script thuyết trình Demo InstOptima (3-5 phút)

> Chạy app: `streamlit run demo_app.py`
> Mục tiêu: đi nhanh qua 3 phần, mỗi phần ~1-1.5 phút.

---

## Mở đầu (10-15 giây)

> "Em xin demo nhanh một ứng dụng minh họa cho InstOptima, gồm 3 phần: Pareto Front, thử Inference, và so sánh hiệu quả thực tế."

---

## Phần 1 — Pareto Front (🏆) ~1 phút

1. Chọn dataset **Laptop14**, Front 1 (Best).
   > "Đây là Pareto Front sau 10 thế hệ NSGA-II — các instruction không bị instruction nào khác lấy đè trên cả 3 mục tiêu Performance, Length, Perplexity."

2. Đổi nhanh **Sort by** sang "Length ↓" rồi "Performance ↑".
   > "Sắp theo từng mục tiêu cho thấy rõ trade-off: instruction ngắn thì Performance có thể thấp hơn một chút, và ngược lại."

3. Mở 1 instruction để show Definition/Example.
   > "Mỗi cá thể gồm Definition và Example — đúng công thức $I = \text{Concat}(d,e)$ đã trình bày."

(Bỏ qua phần Figure 3 nếu thiếu thời gian, hoặc lướt qua 5 giây.)

---

## Phần 2 — Inference (🤖) ~1-1.5 phút

1. Chọn task **SST2**, chọn 1 instruction có Performance cao trong list.
   > "Em chọn 1 instruction từ Pareto Front để thử trực tiếp với FlanT5-base."

2. Dùng câu input mặc định, nhấn **▶ Chạy dự đoán**.
   > "App ghép instruction này với câu input và đưa vào mô hình để sinh dự đoán."

3. Đọc kết quả.
   > "Kết quả: [đọc nhãn]. Đây là minh chứng instruction từ Pareto Front hoạt động đúng qua in-context learning, không cần fine-tune."

> ⚠️ Chạy thử trước để FlanT5-base được cache, tránh chờ lâu khi demo thật.

---

## Phần 3 — So sánh NoInstruct vs InstOptima (📊) ~1-1.5 phút

1. Chọn dataset (ví dụ SST2).
   > "Phần này so sánh 2 mô hình đã fine-tune: một không dùng instruction, một dùng instruction tối ưu từ InstOptima."

2. Đọc nhanh 4 metric ở Mục 1 (đặc biệt chỉ số tăng rõ nhất).
   > "InstOptima cải thiện [đọc 1 số, ví dụ Accuracy +0.0018] so với NoInstruct — đúng với bảng kết quả đã trình bày."

3. Đọc 1 ví dụ ở Mục 3 (NoInstruct sai, InstOptima đúng).
   > "Đây là một ví dụ cụ thể: NoInstruct dự đoán sai, InstOptima dự đoán đúng."

(Bỏ qua Mục 4 — thử trực tiếp 2 mô hình — nếu không đủ thời gian hoặc thiếu checkpoint.)

---

## Kết (10 giây)

> "Như vậy demo đã minh họa từ Pareto Front, đến inference, đến kiểm chứng hiệu quả thực tế của InstOptima. Em xin cảm ơn."

---

## Checklist trước khi demo

- [ ] Chạy `streamlit run demo_app.py`, kiểm tra 3 pickle Pareto load được.
- [ ] Chạy thử Phần 2 1 lần để cache FlanT5-base.
- [ ] Kiểm tra `compare_*.pkl` cho dataset sẽ demo ở Phần 3.
- [ ] Chuẩn bị sẵn việc bỏ qua Figure 3 và Mục 4 nếu thiếu thời gian.
