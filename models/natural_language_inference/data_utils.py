import json

import findfile
import pandas as pd

from .instruction import (
    NLIInstruction,
)


class InstructDatasetLoader:
    def __init__(
        self,
        train_df_id,
        test_df_id,
        train_df_ood=None,
        test_df_ood=None,
        sample_size=1,
    ):
        self.train_df_id = train_df_id.sample(frac=sample_size, random_state=1999)
        self.test_df_id = test_df_id
        if train_df_ood is not None:
            self.train_df_ood = train_df_ood.sample(frac=sample_size, random_state=1999)
        else:
            self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood

    def prepare_instruction_dataloader(self, df, instruction, example):
        """
        Prepare the data in the input format required.
        """
        tc_instructor = NLIInstruction(instruction, example)
        alldata = []
        for i, data in df.iterrows():
            # TC task
            alldata.append(
                {
                    "text": tc_instructor.prepare_input(data["text"]),
                    "labels": data["label"],
                }
            )

        alldata = pd.DataFrame(alldata)
        return alldata

    def create_datasets(self, tokenize_function):
        from datasets import DatasetDict, Dataset

        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if self.test_df_id is None:
            indomain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
        else:
            indomain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_id),
                    "test": Dataset.from_pandas(self.test_df_id),
                }
            )
        indomain_tokenized_datasets = indomain_dataset.map(
            tokenize_function, batched=True
        )

        if (self.train_df_ood is not None) and (self.test_df_ood is None):
            other_domain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {"test": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_ood),
                    "test": Dataset.from_pandas(self.test_df_ood),
                }
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        else:
            other_domain_dataset = None
            other_domain_tokenized_dataset = None

        return (
            indomain_dataset,
            indomain_tokenized_datasets,
            other_domain_dataset,
            other_domain_tokenized_dataset,
        )


import chardet


def read_text(data_path, data_type="train"):
    """
    Sử dụng Hugging Face datasets để tải SNLI thay vì đọc file local bị thiếu.
    """
    from datasets import load_dataset
    
    data = []
    print(f"Đang tải dataset {data_path} từ Hugging Face Hub (split: {data_type})...")
    
    # 1. Tải dataset SNLI
    try:
        # Tải tập snli chuẩn từ Hugging Face
        dataset = load_dataset("snli")
    except Exception as e:
        print(f"Lỗi tải dataset: {e}")
        return []

    # 2. Xử lý chia tách (split)
    if data_type == "train":
        split_data = dataset["train"]
    elif data_type == "test":
        split_data = dataset["test"]
    else:
        split_data = dataset["validation"]
        
    # 3. Chuẩn hóa format
    # Map nhãn số của SNLI sang chữ để khớp với Prompt
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    # Số lượng dữ liệu cần lấy (test nhanh)
    limit = 1000 if data_type == "train" else 500

    for row in split_data:
        label_id = row["label"]
        # Bỏ qua các mẫu có nhãn -1 (thiếu nhãn)
        if label_id in label_map:
            premise = row["premise"].strip()
            hypothesis = row["hypothesis"].strip()
            
            # Ghép 2 câu lại với format giống ví dụ trong Prompt: "Câu 1" "Câu 2"
            text = f'"{premise}" "{hypothesis}"'
            label = label_map[label_id]
            
            data.append({"text": text, "label": label})
            
            # Đủ số lượng thì dừng lại cho nhanh
            if len(data) >= limit:
                break
                
    print(f"Đã tải thành công {len(data)} mẫu cho {data_type}.")
    
    return data
