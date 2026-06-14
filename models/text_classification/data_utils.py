import json

import findfile
import pandas as pd

from .instruction import (
    TCInstruction,
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
        tc_instructor = TCInstruction(instruction, example)
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


def read_text(data_path, data_type="train"):
    """
    Sử dụng Hugging Face datasets để tải SST2 thay vì đọc file local bị thiếu.
    """
    from datasets import load_dataset
    
    data = []
    print(f"Đang tải dataset {data_path} từ Hugging Face Hub (split: {data_type})...")
    
    # 1. Tải dataset SST2 (Glue)
    try:
        dataset = load_dataset("glue", "sst2")
    except Exception as e:
        print(f"Lỗi tải dataset: {e}")
        return []

    # 2. Xử lý chia tách (split)
    # SST2 của glue có các split: 'train', 'validation', 'test'.
    # Tuy nhiên split 'test' không có nhãn (label). Do đó, tác giả thường
    # dùng 'validation' làm test set.
    if data_type == "train":
        split_data = dataset["train"]
    else:
        split_data = dataset["validation"]
        
    # 3. Chuẩn hóa format
    # Theo format của tác giả: data.append({"text": text, "label": label})
    # Label của SST2 gốc: 0 (negative), 1 (positive)
    for row in split_data:
        text = row["sentence"].strip()
        # Chuyển đổi nhãn số thành chuỗi như format cũ mong đợi
        label = "negative" if row["label"] == 0 else "positive"
        data.append({"text": text, "label": label})
        
    # 4. Cắt giảm dữ liệu để chạy thử (giống với ý định cũ của tác giả)
    # Tác giả lấy 1000 mẫu để train cho nhanh.
    # Nếu muốn chạy đầy đủ toàn bộ dataset, bạn thay đổi con số ở đây.
    limit = 1000 if data_type == "train" else 500
    print(f"Đã tải thành công {len(data[:limit])} mẫu cho {data_type}.")
    
    return data[:limit]
