"""
InstOptima Demo — Streamlit app (2-section)
Section 1: Khám phá Instructions Tốt Nhất từ Pareto Front
Section 2: Thử nghiệm Inference với instruction được chọn

Run:  streamlit run demo_app.py
"""
import pickle, re, math, sys, os
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="InstOptima Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Helpers
# ─────────────────────────────────────────────────────────────────────────────

NUM_METRICS = 4
BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ['Laptop14', 'SST2', 'SNLI']
TASK_DESC = {
    'Laptop14': 'Aspect-Based Sentiment Analysis (ABSA)',
    'SST2':     'Text Classification — Sentiment Analysis',
    'SNLI':     'Natural Language Inference (NLI)',
}

# Kết quả so sánh NoInstruct vs InstOptima, sinh bởi compare_noinstruct_vs_instoptima.py
COMPARE_CONFIG = {
    'SST2':     {'pkl': 'compare_tc.pkl',   'ckpt_dir': 'checkpoints_compare',      'task_arg': 'tc'},
    'SNLI':     {'pkl': 'compare_nli.pkl',  'ckpt_dir': 'checkpoints_compare_nli',  'task_arg': 'nli'},
    'Laptop14': {'pkl': 'compare_absa.pkl', 'ckpt_dir': 'checkpoints_compare_absa', 'task_arg': 'absa'},
}

def resolve_path(rel_path):
    """Tìm file/thư mục theo rel_path trong BASE, hoặc cùng đường dẫn ở các ổ đĩa khác
    (script fine-tune có thể đã chạy và lưu kết quả ở ổ đĩa khác với demo_app.py)."""
    drive, tail = os.path.splitdrive(BASE)
    candidates = [os.path.join(BASE, rel_path)]
    for d in ('D:', 'F:', 'C:', 'E:'):
        if d.upper() != drive.upper():
            candidates.append(os.path.join(d + tail, rel_path))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def parse_obj(ind):
    try:
        return float(ind.objectives[0]), float(ind.objectives[1]), float(ind.objectives[2])
    except Exception:
        s = str(getattr(ind, 'objectives', ''))
        lm = re.search(r'instruction_length=([0-9.]+)', s)
        pm = re.search(r'perplexity=([0-9.]+)', s)
        rm = re.search(r'reciprocal_metric=([0-9.]+)', s)
        if lm and pm and rm:
            return float(lm.group(1)), float(pm.group(1)), float(rm.group(1))
    return None, None, None

def recip_to_avg(r):
    return (1.0 / r) / NUM_METRICS if r and not math.isinf(r) else 0.0

@st.cache_resource
def load_data():
    noi   = pickle.load(open(os.path.join(BASE, 'noinstruct_all.pkl'),  'rb'))
    ran   = pickle.load(open(os.path.join(BASE, 'raninstruct_all.pkl'), 'rb'))
    nsga2 = {
        'Laptop14': pickle.load(open(os.path.join(BASE, 'nsga2_result_v4.pkl'),  'rb')),
        'SST2':     pickle.load(open(os.path.join(BASE, 'nsga2_result_tc.pkl'),   'rb')),
        'SNLI':     pickle.load(open(os.path.join(BASE, 'nsga2_result_nli.pkl'),  'rb')),
    }
    return noi, ran, nsga2

@st.cache_resource
def load_model():
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    model_name = "google/flan-t5-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_compare_data(dataset_key):
    cfg = COMPARE_CONFIG[dataset_key]
    path = resolve_path(cfg['pkl'])
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, 'rb'))

@st.cache_resource
def load_finetuned_models(dataset_key):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    cfg = COMPARE_CONFIG[dataset_key]
    base_dir = resolve_path(cfg['ckpt_dir'])
    models = {}
    for name in ['noinstruct', 'instoptima']:
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(path)
            mdl.eval()
            models[name] = (tok, mdl)
    return models

def get_instructor(dataset_key, definition, example):
    if dataset_key == 'SST2':
        from models.text_classification.instruction import TCInstruction
        return TCInstruction(definition, example)
    elif dataset_key == 'SNLI':
        from models.natural_language_inference.instruction import NLIInstruction
        return NLIInstruction(definition, example)
    else:
        from models.aspect_based_sentiment_analysis.instruction import APCInstruction
        return APCInstruction(definition, example)

def get_valid_front(nsga2_data, dataset, front_id=0):
    fronts = nsga2_data[dataset].get('fronts', [])
    if front_id >= len(fronts):
        return []
    valid = []
    for ind in fronts[front_id]:
        l, p, r = parse_obj(ind)
        if l and p and r and not math.isinf(r):
            valid.append((ind, l, p, r, recip_to_avg(r)))
    return valid

ABSA_BALANCED_EXAMPLES = """\
The screen is beautiful but the battery drains too fast.
The sentiments of screen | battery are:
screen:positive | battery:negative

The keyboard is awful and the price is also way too expensive.
The sentiments of keyboard | price are:
keyboard:negative | price:negative

The camera quality is outstanding and the design looks sleek.
The sentiments of camera | design are:
camera:positive | design:positive"""

SNLI_BALANCED_EXAMPLES = """\
input: "A man is playing guitar on stage." "A person is performing music."
Your judgment is:
entailment

input: "A woman is cooking dinner in the kitchen." "The woman is sleeping in her bed."
Your judgment is:
contradiction

input: "A child is riding a bicycle on the sidewalk." "The child is going to school."
Your judgment is:
neutral"""

def build_prompt(task, defn, ex, user_inputs):
    ex_text = str(ex).strip() if ex else ""
    if task == "Laptop14":
        text    = user_inputs.get("text", "")
        aspects = user_inputs.get("aspects", "")
        query   = f"{text}\nThe sentiments of {aspects} are:"
    elif task == "SST2":
        text  = user_inputs.get("text", "")
        query = f"input: {text}\nThe statement is:"
    else:  # SNLI
        premise    = user_inputs.get("premise", "")
        hypothesis = user_inputs.get("hypothesis", "")
        query = f'input: "{premise}" "{hypothesis}"\nYour judgment is:'

    parts = [defn.strip()]
    if ex_text:
        parts.append(ex_text)
    if task == "Laptop14":
        parts.append(ABSA_BALANCED_EXAMPLES)
    elif task == "SNLI":
        parts.append(SNLI_BALANCED_EXAMPLES)
    parts.append(query)
    return "\n\n".join(parts)

def run_inference(tokenizer, model, prompt, max_new_tokens=128, decoder_prefix=None):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens)
    if decoder_prefix:
        prefix_ids = tokenizer(decoder_prefix, return_tensors="pt", add_special_tokens=False).input_ids
        pad_id = torch.tensor([[model.config.decoder_start_token_id]])
        gen_kwargs["decoder_input_ids"] = torch.cat([pad_id, prefix_ids], dim=1)
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

_, _, nsga2_data = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 InstOptima")
    st.markdown("**Evolutionary Multi-objective\nInstruction Optimization**")
    st.markdown("---")
    section = st.radio("Chọn phần", [
        "🏆 Khám phá Pareto Front",
        "🤖 Thử nghiệm Inference",
        "📊 So sánh NoInstruct vs InstOptima",
    ])
    st.markdown("---")
    st.caption("EMNLP 2023 · Yang & Li")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Pareto Explorer
# ─────────────────────────────────────────────────────────────────────────────

if section == "🏆 Khám phá Pareto Front":
    st.title("🏆 Khám phá Instructions Tốt Nhất từ Pareto Front")

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        ds_best  = st.selectbox("Dataset", DATASETS, key="best_ds")
        front_id = st.selectbox(
            "Pareto Front", [0, 1, 2],
            format_func=lambda x: f"Front {x+1} ({'Best' if x==0 else 'Good' if x==1 else 'OK'})",
        )
        sort_by = st.radio("Sort by", ["Performance ↑", "Length ↓", "Perplexity ↓"])
        top_n   = st.slider("Hiển thị top N", 1, 20, 5)

    valid = get_valid_front(nsga2_data, ds_best, front_id)

    if not valid:
        st.warning(f"Không có dữ liệu cho {ds_best} Front {front_id+1}.")
    else:
        if sort_by == "Performance ↑":
            valid.sort(key=lambda x: x[4], reverse=True)
        elif sort_by == "Length ↓":
            valid.sort(key=lambda x: x[1])
        else:
            valid.sort(key=lambda x: x[2])

        with col_main:
            st.markdown(
                f"**{ds_best} — {TASK_DESC[ds_best]}**  \n"
                f"Front {front_id+1} · {len(valid)} instructions"
            )
            for rank, (ind, l, p, r, avg) in enumerate(valid[:top_n]):
                instr = getattr(ind, 'instruction', None)
                defn  = getattr(instr, 'definition', str(instr)) if instr else '(no instruction)'
                ex    = getattr(instr, 'example', '')            if instr else ''

                with st.expander(f"#{rank+1}  |  Avg={avg:.4f}  |  Len={l:.0f}  |  Ppl={p:.4f}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Performance", f"{avg:.4f}")
                    c2.metric("Length",      f"{l:.0f}")
                    c3.metric("Perplexity",  f"{p:.4f}")
                    st.markdown("**Definition:**")
                    st.markdown(f"> {defn}")
                    if ex:
                        st.markdown("**Example:**")
                        st.code(str(ex)[:600] + ('...' if len(str(ex)) > 600 else ''), language='text')

    st.divider()
    pareto_path = os.path.join(BASE, 'paper_fig3_pareto.png')
    if os.path.exists(pareto_path):
        st.subheader("Figure 3: 2D Pareto Fronts")
        st.image(pareto_path, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Inference
# ─────────────────────────────────────────────────────────────────────────────

elif section == "🤖 Thử nghiệm Inference":
    st.title("🤖 Thử nghiệm: Chọn Instruction và Dự đoán")

    # ── Step 1: chọn task ──────────────────────────────────────────────────
    st.subheader("Bước 1 — Chọn Task")
    task = st.selectbox("Task", DATASETS, format_func=lambda x: f"{x} — {TASK_DESC[x]}")

    # ── Step 2: chọn instruction từ Pareto Front ──────────────────────────
    st.subheader("Bước 2 — Chọn Instruction từ Pareto Front")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        sort_inf = st.radio(
            "Sort instructions by",
            ["Performance ↑", "Length ↓", "Perplexity ↓"],
            key="inf_sort",
        )
        top_inf = st.slider("Số instruction hiển thị", 3, 20, 10, key="inf_top")

    valid_inf = get_valid_front(nsga2_data, task, front_id=0)
    if sort_inf == "Performance ↑":
        valid_inf.sort(key=lambda x: x[4], reverse=True)
    elif sort_inf == "Length ↓":
        valid_inf.sort(key=lambda x: x[1])
    else:
        valid_inf.sort(key=lambda x: x[2])

    valid_inf = valid_inf[:top_inf]

    # Build label list for selectbox
    def instr_label(rank, ind, l, p, avg):
        instr = getattr(ind, 'instruction', None)
        defn  = getattr(instr, 'definition', str(instr)) if instr else ''
        snippet = defn[:70].replace('\n', ' ') + ('…' if len(defn) > 70 else '')
        return f"#{rank+1} | Avg={avg:.4f} | Len={l:.0f} | Ppl={p:.4f} | {snippet}"

    options = [instr_label(i, ind, l, p, avg) for i, (ind, l, p, r, avg) in enumerate(valid_inf)]

    with col_s2:
        chosen_label = st.selectbox("Chọn instruction", options, key="chosen_instr")

    chosen_idx = options.index(chosen_label)
    chosen_ind, chosen_l, chosen_p, chosen_r, chosen_avg = valid_inf[chosen_idx]
    chosen_instr = getattr(chosen_ind, 'instruction', None)
    chosen_defn  = getattr(chosen_instr, 'definition', str(chosen_instr)) if chosen_instr else ''
    chosen_ex    = getattr(chosen_instr, 'example', '')                    if chosen_instr else ''

    with st.expander("📄 Xem instruction được chọn", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Performance", f"{chosen_avg:.4f}")
        c2.metric("Length",      f"{chosen_l:.0f}")
        c3.metric("Perplexity",  f"{chosen_p:.4f}")
        st.markdown("**Definition:**")
        st.markdown(f"> {chosen_defn}")
        if chosen_ex:
            st.markdown("**Example:**")
            st.code(str(chosen_ex)[:600], language='text')

    # ── Step 3: nhập input ────────────────────────────────────────────────
    st.subheader("Bước 3 — Nhập Input")

    user_inputs = {}

    if task == "Laptop14":
        st.markdown("Task: **Aspect-Based Sentiment Analysis** — Phân tích cảm xúc theo từng khía cạnh")
        user_inputs["text"] = st.text_area(
            "Câu đánh giá sản phẩm/dịch vụ",
            value="The battery life is great but the keyboard feels a bit stiff.",
            height=80,
        )
        user_inputs["aspects"] = st.text_input(
            "Các khía cạnh cần phân tích (cách nhau bởi |)",
            value="battery life | keyboard",
        )

    elif task == "SST2":
        st.markdown("Task: **Text Classification** — Phân loại cảm xúc (positive / negative)")
        user_inputs["text"] = st.text_area(
            "Câu cần phân loại",
            value="This movie was absolutely wonderful and I loved every minute of it.",
            height=80,
        )

    else:  # SNLI
        st.markdown("Task: **Natural Language Inference** — Phân loại quan hệ (entailment / contradiction / neutral)")
        user_inputs["premise"] = st.text_area(
            "Premise (câu giả thiết)",
            value="A man is playing a guitar on stage in front of a large crowd.",
            height=70,
        )
        user_inputs["hypothesis"] = st.text_area(
            "Hypothesis (câu cần suy diễn)",
            value="A person is performing music live.",
            height=70,
        )

    # ── Step 4: chạy inference ────────────────────────────────────────────
    st.subheader("Bước 4 — Chạy Dự đoán")

    prompt_preview = build_prompt(task, chosen_defn, chosen_ex, user_inputs)

    if st.button("▶ Chạy dự đoán", type="primary", use_container_width=True):
        with st.spinner("Đang load FlanT5-base và chạy inference..."):
            try:
                tokenizer, model = load_model()
                if task == "Laptop14":
                    aspects_list = [a.strip() for a in user_inputs.get("aspects", "").split("|")]
                    parts = []
                    for aspect in aspects_list:
                        single_inputs = dict(user_inputs, aspects=aspect)
                        single_prompt = build_prompt(task, chosen_defn, chosen_ex, single_inputs)
                        pred = run_inference(tokenizer, model, single_prompt, decoder_prefix=f"{aspect}:")
                        if ":" in pred:
                            sentiment = pred.split(":", 1)[1].strip().split("|")[0].strip().split("\n")[0].strip()
                        else:
                            sentiment = pred.strip()
                        parts.append(f"{aspect}:{sentiment}")
                    result = " | ".join(parts)
                else:
                    result = run_inference(tokenizer, model, prompt_preview)
            except Exception as e:
                st.error(f"Lỗi khi chạy inference: {e}")
                result = None

        if result is not None:
            st.success("✅ Dự đoán hoàn tất!")
            st.markdown("### Kết quả dự đoán")

            if task == "Laptop14":
                st.info(f"**Sentiment của các aspects:** `{result}`")
                st.caption(f"Aspects: `{user_inputs.get('aspects', '')}`  |  Input: *{user_inputs.get('text', '')[:80]}...*")
            elif task == "SST2":
                label = result.strip().lower()
                color = "🟢 **Positive**" if "positive" in label else ("🔴 **Negative**" if "negative" in label else f"⚪ **{result}**")
                st.info(f"Phân loại cảm xúc: {color}")
                st.caption(f"Input: *{user_inputs.get('text', '')[:100]}*")
            else:
                label = result.strip().lower()
                if "entailment" in label:
                    display = "✅ **Entailment** — Hypothesis được suy ra từ Premise"
                elif "contradiction" in label:
                    display = "❌ **Contradiction** — Hypothesis mâu thuẫn với Premise"
                else:
                    display = "⚪ **Neutral** — Không có quan hệ rõ ràng"
                st.info(display)
                st.caption(
                    f"Premise: *{user_inputs.get('premise', '')[:60]}*  \n"
                    f"Hypothesis: *{user_inputs.get('hypothesis', '')[:60]}*"
                )

            st.markdown("---")
            st.markdown(f"**Raw output:** `{result}`")
            st.markdown(f"**Instruction dùng:** Avg={chosen_avg:.4f} | Len={chosen_l:.0f} | Ppl={chosen_p:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — So sánh NoInstruct vs InstOptima (model đã fine-tune thật)
# ─────────────────────────────────────────────────────────────────────────────

elif section == "📊 So sánh NoInstruct vs InstOptima":
    st.title("📊 So sánh: Không dùng Instruction vs Instruction tối ưu từ InstOptima")
    st.caption(
        "Dữ liệu được tạo bằng `compare_noinstruct_vs_instoptima.py`: fine-tune 2 mô hình "
        "FlanT5-base trên cùng tập dữ liệu — một với instruction rỗng (NoInstruct), "
        "một với instruction tốt nhất tìm được bởi InstOptima."
    )

    compare_dataset = st.selectbox("Chọn dataset", DATASETS, key="compare_dataset")
    cfg = COMPARE_CONFIG[compare_dataset]
    compare_data = load_compare_data(compare_dataset)
    if compare_data is None:
        st.warning(
            f"Chưa tìm thấy `{cfg['pkl']}`. Hãy chạy trước:\n\n"
            f"```\npython compare_noinstruct_vs_instoptima.py --task {cfg['task_arg']} --epochs 3 --num_compare 30\n```"
        )
    else:
        st.markdown(f"**Dataset:** {compare_data['dataset']}  ·  **PLM:** {compare_data['plm']}")

        st.subheader("1. Hiệu suất sau khi fine-tune trên tập test")
        m_no  = compare_data['noinstruct_metrics']
        m_opt = compare_data['instoptima_metrics']
        cols = st.columns(4)
        for col, key, label in zip(cols, ['accuracy', 'precision', 'recall', 'f1'], ['Accuracy', 'Precision', 'Recall', 'F1']):
            delta = m_opt[key] - m_no[key]
            col.metric(label, f"{m_opt[key]:.4f}", delta=f"{delta:+.4f}", help=f"NoInstruct: {m_no[key]:.4f}")

        st.subheader("2. Instruction tối ưu (InstOptima) đã dùng")
        instr = compare_data['instoptima_instruction']
        l, p, r = instr['objectives']
        c1, c2, c3 = st.columns(3)
        c1.metric("Length", f"{l:.0f}")
        c2.metric("Perplexity", f"{p:.4f}")
        c3.metric("Avg Performance", f"{recip_to_avg(r):.4f}")
        st.markdown("**Definition:**")
        st.markdown(f"> {instr['definition']}")
        if instr['example']:
            st.markdown("**Example:**")
            st.code(str(instr['example'])[:600], language='text')

        st.subheader("3. Ví dụ: NoInstruct dự đoán SAI nhưng InstOptima dự đoán ĐÚNG")
        showcase = compare_data['showcase']
        if not showcase:
            st.info("Không tìm thấy ví dụ nào trong tập test.")
        else:
            for r_ex in showcase:
                with st.container(border=True):
                    st.markdown(f"*{r_ex['text']}*")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Nhãn đúng:** `{r_ex['true_label']}`")
                    c2.markdown(f"❌ **NoInstruct:** `{r_ex['noinstruct_pred']}`")
                    c3.markdown(f"✅ **InstOptima:** `{r_ex['instoptima_pred']}`")

        st.divider()
        st.subheader("4. Thử trực tiếp với 2 mô hình đã fine-tune")
        models = load_finetuned_models(compare_dataset)
        if 'noinstruct' not in models or 'instoptima' not in models:
            st.info(
                f"Không tìm thấy checkpoint trong `{cfg['ckpt_dir']}/`. "
                "Phần thử trực tiếp chỉ khả dụng trên máy đã chạy script fine-tune."
            )
        else:
            no_instr  = get_instructor(compare_dataset, "", "")
            opt_instr = get_instructor(compare_dataset, instr['definition'], instr['example'])

            if compare_dataset == "SST2":
                default_text = showcase[0]['text'] if showcase else "This movie was absolutely wonderful."
                text_input = st.text_area("Câu cần phân loại", value=default_text, height=80)

                if st.button("▶ So sánh dự đoán", type="primary"):
                    no_prompt  = no_instr.prepare_input(text_input)
                    opt_prompt = opt_instr.prepare_input(text_input)

                    no_tok, no_model   = models['noinstruct']
                    opt_tok, opt_model = models['instoptima']
                    with st.spinner("Đang chạy 2 mô hình..."):
                        no_pred  = run_inference(no_tok, no_model, no_prompt, max_new_tokens=8)
                        opt_pred = run_inference(opt_tok, opt_model, opt_prompt, max_new_tokens=8)

                    c1, c2 = st.columns(2)
                    c1.metric("NoInstruct dự đoán", no_pred.strip())
                    c2.metric("InstOptima dự đoán", opt_pred.strip())

            elif compare_dataset == "SNLI":
                default_premise, default_hypothesis = "A man is playing guitar on stage.", "A person is performing music."
                if showcase:
                    m = re.match(r'"(.*)"\s+"(.*)"', showcase[0]['text'])
                    if m:
                        default_premise, default_hypothesis = m.group(1), m.group(2)
                premise = st.text_input("Premise", value=default_premise)
                hypothesis = st.text_input("Hypothesis", value=default_hypothesis)

                if st.button("▶ So sánh dự đoán", type="primary"):
                    text_input = f'"{premise}" "{hypothesis}"'
                    no_prompt  = no_instr.prepare_input(text_input)
                    opt_prompt = opt_instr.prepare_input(text_input)

                    no_tok, no_model   = models['noinstruct']
                    opt_tok, opt_model = models['instoptima']
                    with st.spinner("Đang chạy 2 mô hình..."):
                        no_pred  = run_inference(no_tok, no_model, no_prompt, max_new_tokens=8)
                        opt_pred = run_inference(opt_tok, opt_model, opt_prompt, max_new_tokens=8)

                    c1, c2 = st.columns(2)
                    c1.metric("NoInstruct dự đoán", no_pred.strip())
                    c2.metric("InstOptima dự đoán", opt_pred.strip())

            else:  # Laptop14
                default_text = showcase[0]['text'] if showcase else "The battery life is great but the keyboard is uncomfortable."
                text_input = st.text_area("Câu (review)", value=default_text, height=80)
                aspects_input = st.text_input("Aspects (phân cách bởi `|`)", value="battery life | keyboard")

                if st.button("▶ So sánh dự đoán", type="primary"):
                    no_prompt  = no_instr.prepare_input(text_input, aspects_input)
                    opt_prompt = opt_instr.prepare_input(text_input, aspects_input)

                    no_tok, no_model   = models['noinstruct']
                    opt_tok, opt_model = models['instoptima']
                    with st.spinner("Đang chạy 2 mô hình..."):
                        no_pred  = run_inference(no_tok, no_model, no_prompt, max_new_tokens=64)
                        opt_pred = run_inference(opt_tok, opt_model, opt_prompt, max_new_tokens=64)

                    c1, c2 = st.columns(2)
                    c1.metric("NoInstruct dự đoán", no_pred.strip())
                    c2.metric("InstOptima dự đoán", opt_pred.strip())
