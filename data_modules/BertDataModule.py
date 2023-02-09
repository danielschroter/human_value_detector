
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
import torch



class BertDataModule(pl.LightningDataModule):

    def __init__(self, train_df, val_df, tokenizer, params, label_columns):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.params = params
        self.tokenizer = tokenizer
        self.label_columns = label_columns


    def setup(self, stage=None):
        self.train_dataset = BertDataset(
            data = self.train_df,
            tokenizer=self.tokenizer,
            max_token_count=self.params["MAX_TOKEN_COUNT"],
            label_columns = self.label_columns,
        )

        self.val_dataset = BertDataset(
            data = self.val_df,
            tokenizer=self.tokenizer,
            max_token_count=self.params["MAX_TOKEN_COUNT"],
            label_columns = self.label_columns,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.params["NUM_TRAIN_WORKERS"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params["BATCH_SIZE"],
            num_workers=self.params["NUM_VAL_WORKERS"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params["BATCH_SIZE"],
            num_workers=self.params["NUM_VAL_WORKERS"]
        )


class BertDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            max_token_count,
            label_columns=None,

    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_count
        if label_columns:
            self.label_columns = label_columns
        else:
            self.label_columns = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row.text
        if self.label_columns:
            labels = data_row[self.label_columns]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.label_columns:
            return dict(
                text=text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                labels=torch.FloatTensor(labels)
            )
        else:
            return dict(
                text=text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten()
            )




#%%

