# InstOptima — CS410.Q21 Nhóm 8 — Đồ án cuối kỳ

Source code cho đồ án dựa trên bài báo **InstOptima: Evolutionary Multi-objective Instruction Optimization via
Large Language Model-based Instruction Operators** (EMNLP 2023 Findings).

## 1. Cấu trúc source code nộp

```
InstOptima/
├── README.md          # file này
├── requirements.txt
│
├── chatgpt.py                     # Wrapper gọi ChatGPT để mô phỏng instruction operators
├── main.py                        # Chạy NSGA-II cho task ABSA (Laptop14)
├── main_tc.py                     # Chạy NSGA-II cho task Text Classification (SST2)
├── main_nli.py                    # Chạy NSGA-II cho task NLI (SNLI)
│
├── entity/                        # Individual, Instruction, Population
├── evo_core/                      # Thuật toán NSGA-II
├── operators/                     # Instruction/Prompt operators (mutation, crossover)
├── objectives/                    # Tính 3 mục tiêu: performance (m), length (l), perplexity (r)
├── models/                        # Model + data_utils cho từng task (ABSA, TC, NLI, Sum)
├── datasets/                      # Dữ liệu Laptop14 / SST2 / SNLI (đã rút gọn 1000/1000/1000)
│
├── run_noinstruct.py              # Baseline "NoInstruct" (không dùng instruction)
├── run_raninstruct.py             # Baseline "RanInstruct" (5 instruction sinh ngẫu nhiên bằng ChatGPT)
├── compare_noinstruct_vs_instoptima.py  # Fine-tune & so sánh NoInstruct vs InstOptima
├── get_best_prompts.py            # Trích xuất top instruction từ Pareto front
├── analyze_results.py             # Phân tích nhanh nội dung 1 file pkl kết quả
│
├── plot_pareto.py                 # Vẽ Figure 3 - Pareto front 2D
├── plot_trajectory.py             # Vẽ Figure 2 - trajectory qua các thế hệ
├── plot_evolution_gif.py          # Tạo GIF minh họa quá trình tiến hóa
├── plot_comparison_table.py       # Vẽ bảng/biểu đồ so sánh Table 1
│
├── demo_app.py                    # Demo Streamlit (3 phần: Pareto / Inference / So sánh)
│
├── nsga2_result_v4.pkl            # Kết quả NSGA-II - Laptop14 (ABSA)
├── nsga2_result_tc.pkl            # Kết quả NSGA-II - SST2 (TC)
├── nsga2_result_nli.pkl           # Kết quả NSGA-II - SNLI (NLI)
├── noinstruct_*.pkl                # Kết quả baseline NoInstruct (3 dataset + tổng hợp)
├── raninstruct_*.pkl               # Kết quả baseline RanInstruct (3 dataset + tổng hợp)
└──compare_tc.pkl                  # Kết quả so sánh fine-tune NoInstruct vs InstOptima (SST2)
```

## 2. Cài đặt môi trường

```bash
conda create -n instoptima python=3.9
conda activate instoptima
pip install -r requirements.txt
```

Một số script (`run_raninstruct.py`, các instruction operator trong `chatgpt.py`) cần gọi OpenAI API.
Đặt API key qua biến môi trường (không hardcode trong code):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

## 3. Cách chạy

### 3.1. Chạy tối ưu hóa NSGA-II (sinh Pareto front)

```bash
python main.py       # Laptop14 (ABSA)   -> nsga2_result_v4.pkl
python main_tc.py    # SST2 (TC)         -> nsga2_result_tc.pkl
python main_nli.py   # SNLI (NLI)        -> nsga2_result_nli.pkl
```

Cấu hình mặc định: population = 30, generations = 10, PLM = `google/flan-t5-base`.

### 3.2. Chạy baseline để so sánh

```bash
python run_noinstruct.py    # NoInstruct trên cả 3 dataset -> noinstruct_*.pkl
python run_raninstruct.py   # RanInstruct (5 instruction random qua ChatGPT) -> raninstruct_*.pkl
```

### 3.3. Fine-tune & so sánh NoInstruct vs InstOptima (cho demo phần 3)

```bash
python compare_noinstruct_vs_instoptima.py --task tc --epochs 3 --num_compare 30
python compare_noinstruct_vs_instoptima.py --task nli --epochs 3 --num_compare 30
python compare_noinstruct_vs_instoptima.py --task absa --epochs 3 --num_compare 30
```

Script này fine-tune 2 mô hình FlanT5-base (NoInstruct vs instruction tốt nhất từ Pareto front của
`nsga2_result_*.pkl`), lưu checkpoint vào `checkpoints_compare*/` và kết quả so sánh vào `compare_*.pkl`.

### 3.4. Phân tích / trích xuất kết quả

```bash
python get_best_prompts.py     # in top-5 instruction tốt nhất từ 1 file pkl (sửa file_path trong script)
python analyze_results.py      # phân tích nhanh nội dung các file pkl
```

### 3.5. Vẽ lại các hình trong slide

```bash
python plot_pareto.py            # -> paper_fig3_pareto.png
python plot_trajectory.py        # -> paper_fig2_trajectory.png
python plot_evolution_gif.py     # -> paper_evolution.gif
python plot_comparison_table.py  # -> paper_table1_*.png
```

### 3.6. Chạy Demo

```bash
streamlit run demo_app.py
```

Demo gồm 3 phần:
1. **Pareto Front Explorer** — duyệt các instruction tối ưu theo Performance/Length/Perplexity.
2. **Inference** — chọn 1 instruction từ Pareto front, nhập câu mới và chạy dự đoán với FlanT5-base.
3. **So sánh NoInstruct vs InstOptima** — số liệu fine-tune thực tế (yêu cầu đã chạy bước 3.3 để có
   `compare_*.pkl` và checkpoint tương ứng; nếu thiếu, demo sẽ tự ẩn phần thử trực tiếp).

Lần đầu chạy sẽ tự tải model `google/flan-t5-base` từ HuggingFace (cần kết nối Internet).

## 4. Ghi chú

- Bộ dữ liệu rút gọn: mỗi dataset (Laptop14, SST2, SNLI) dùng 1000 mẫu train / 1000 validation / 1000 test
  để giảm chi phí tính toán trong quá trình tiến hóa.
- 3 mục tiêu tối ưu $F=(m,l,r)$: $m$ = nghịch đảo tổng (Accuracy+Precision+Recall+F1), $l$ = số ký tự
  instruction, $r$ = perplexity (RoBERTa masked-LM).
- Repo gốc của bài báo: https://github.com/yangheng95/InstOptima
