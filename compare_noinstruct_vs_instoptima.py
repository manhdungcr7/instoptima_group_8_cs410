"""
Fine-tune two FlanT5-base checkpoints - một dùng "NoInstruct" (instruction rỗng)
và một dùng instruction tốt nhất do InstOptima tìm được (Pareto front, hiệu suất cao nhất)
- trên cùng một tập dữ liệu, sau đó so sánh dự đoán của 2 mô hình trên cùng tập test
để tìm ra các ví dụ mà NoInstruct trả lời SAI nhưng InstOptima trả lời ĐÚNG.

Kết quả được lưu vào 1 file pkl để demo_app.py có thể đọc và hiển thị so sánh.

Cách chạy (trong môi trường Anaconda đã cài transformers/datasets/torch/...):

    python compare_noinstruct_vs_instoptima.py --task tc --epochs 3 --num_compare 30
    python compare_noinstruct_vs_instoptima.py --task nli --epochs 3 --num_compare 30

Lưu ý: khác với train_tc/train_nli gốc, script này KHÔNG xóa thư mục checkpoints
sau khi train, vì checkpoint sẽ được dùng lại cho demo.
"""
import argparse
import importlib
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd


TASK_CONFIG = {
    "tc": {
        "module": "models.text_classification",
        "pareto_pkl": "nsga2_result_tc.pkl",
        "dataset": "SST2",
        "read_fn": "read_text",
    },
    "nli": {
        "module": "models.natural_language_inference",
        "pareto_pkl": "nsga2_result_nli.pkl",
        "dataset": "SNLI",
        "read_fn": "read_text",
    },
    "absa": {
        "module": "models.aspect_based_sentiment_analysis",
        "pareto_pkl": "nsga2_result_v4.pkl",
        "dataset": "Laptop14",
        "read_fn": "read_json",
    },
}


def get_best_individual(pkl_path):
    """Đọc kết quả NSGA-II và lấy cá thể có hiệu suất (m) tốt nhất trong Pareto front 0."""
    with open(pkl_path, "rb") as f:
        evo_data = pickle.load(f)
    front0 = evo_data["fronts"][0]
    # objectives.objectives = [length, perplexity, 1/(acc+prec+rec+f1)] -> nhỏ hơn = hiệu suất cao hơn
    best = min(front0, key=lambda ind: ind.objectives.objectives[2])
    return best


def run_config(name, instruction, example, plm, dataset, epochs, module, output_dir, read_fn="read_text"):
    model_mod = importlib.import_module(f"{module}.model")
    data_mod = importlib.import_module(f"{module}.data_utils")

    T5Generator = model_mod.T5Generator
    InstructDatasetLoader = data_mod.InstructDatasetLoader
    read_data = getattr(data_mod, read_fn)

    id_tr_df = pd.DataFrame(read_data(dataset, "train"))
    id_te_df = pd.DataFrame(read_data(dataset, "test"))

    loader = InstructDatasetLoader(id_tr_df, id_te_df)
    loader.train_df_id = loader.prepare_instruction_dataloader(
        loader.train_df_id, instruction, example
    )
    loader.test_df_id = loader.prepare_instruction_dataloader(
        loader.test_df_id, instruction, example
    )

    t5_exp = T5Generator(plm)
    id_ds, id_tokenized_ds, _, _ = loader.create_datasets(t5_exp.tokenize_function_inputs)

    model_out_path = os.path.join(output_dir, name)
    training_args = {
        "output_dir": model_out_path,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "load_best_model_at_end": True,
        "push_to_hub": False,
        "eval_accumulation_steps": 1,
        "predict_with_generate": True,
        "logging_steps": 1000000000,
        "use_mps_device": False,
        "logging_strategy": "no",
        "disable_tqdm": True,
    }

    print(f"\n>>> [{name}] Bắt đầu fine-tune {plm} trên {dataset} ({epochs} epochs)...")
    trainer = t5_exp.train(id_tokenized_ds, **training_args)

    pred_labels = t5_exp.get_labels(
        predictor=trainer, tokenized_dataset=id_tokenized_ds, sample_set="test", batch_size=4
    )
    true_labels = [i.strip() for i in id_ds["test"]["labels"]]

    metrics = t5_exp.get_classic_metrics(list(true_labels), list(pred_labels))
    print(f">>> [{name}] metrics trên test set: {metrics}")

    raw_test = id_te_df.to_dict("records")  # {"text", "label"} gốc, chưa gắn instruction

    return {
        "metrics": metrics,
        "pred_labels": [p.replace(" ", "") for p in pred_labels],
        "true_labels": [t.replace(" ", "") for t in true_labels],
        "raw_test": raw_test,
        "model_path": model_out_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tc", "nli", "absa"], default="tc")
    parser.add_argument("--plm", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_compare", type=int, default=30)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--out_pkl", default=None)
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    # Mỗi task dùng một thư mục checkpoint riêng để không ghi đè lẫn nhau
    output_dir = args.output_dir or (
        "checkpoints_compare" if args.task == "tc" else f"checkpoints_compare_{args.task}"
    )

    print(f"Đang đọc Pareto front từ {cfg['pareto_pkl']} ...")
    best = get_best_individual(cfg["pareto_pkl"])
    instoptima_definition = best.instruction.definition
    instoptima_example = best.instruction.example
    instoptima_objectives = best.objectives.objectives
    instoptima_metric = best.objectives.metric

    print("Instruction (definition) tốt nhất từ InstOptima:")
    print(instoptima_definition)
    print("Objectives F=(length, perplexity, 1/sum_metrics):", instoptima_objectives)
    print("Metric gốc:", instoptima_metric)

    os.makedirs(output_dir, exist_ok=True)

    no_instruct = run_config(
        "noinstruct", "", "", args.plm, cfg["dataset"], args.epochs, cfg["module"], output_dir,
        read_fn=cfg["read_fn"],
    )
    instoptima = run_config(
        "instoptima",
        instoptima_definition,
        instoptima_example,
        args.plm,
        cfg["dataset"],
        args.epochs,
        cfg["module"],
        output_dir,
        read_fn=cfg["read_fn"],
    )

    # Test set giống hệt nhau (cùng dataset, cùng thứ tự) nên có thể so sánh theo từng index
    raw_test = no_instruct["raw_test"]
    n = min(len(raw_test), len(no_instruct["pred_labels"]), len(instoptima["pred_labels"]))

    records = []
    for i in range(n):
        records.append(
            {
                "text": raw_test[i]["text"],
                "true_label": no_instruct["true_labels"][i],
                "noinstruct_pred": no_instruct["pred_labels"][i],
                "instoptima_pred": instoptima["pred_labels"][i],
            }
        )

    showcase = [
        r
        for r in records
        if r["noinstruct_pred"] != r["true_label"] and r["instoptima_pred"] == r["true_label"]
    ]
    print(f"\nTìm thấy {len(showcase)} ví dụ NoInstruct SAI nhưng InstOptima ĐÚNG.")
    for r in showcase[: args.num_compare]:
        print("-" * 80)
        print("Text:", r["text"])
        print("True:", r["true_label"], "| NoInstruct:", r["noinstruct_pred"], "| InstOptima:", r["instoptima_pred"])

    result = {
        "task": args.task,
        "dataset": cfg["dataset"],
        "plm": args.plm,
        "instoptima_instruction": {
            "definition": instoptima_definition,
            "example": instoptima_example,
            "objectives": instoptima_objectives,
            "metric": instoptima_metric,
        },
        "noinstruct_metrics": no_instruct["metrics"],
        "instoptima_metrics": instoptima["metrics"],
        "records": records,
        "showcase": showcase[: args.num_compare],
        "model_paths": {
            "noinstruct": no_instruct["model_path"],
            "instoptima": instoptima["model_path"],
        },
    }

    out_pkl = args.out_pkl or f"compare_{args.task}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"\nĐã lưu kết quả so sánh vào {out_pkl}")
    print(f"Checkpoint NoInstruct: {no_instruct['model_path']}")
    print(f"Checkpoint InstOptima: {instoptima['model_path']}")


if __name__ == "__main__":
    main()
