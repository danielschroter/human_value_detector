import torch
import pytorch_lightning as pl

import numpy as np

from transformers import (
    get_linear_schedule_with_warmup,
    AutoModel
)
import torch.nn as nn
import torch.functional as F
from torchmetrics import AUROC, F1Score
from torch.optim import AdamW

from sklearn.metrics import classification_report
from toolbox.bert_utils import max_for_thres


class BertFineTunerPl(pl.LightningModule):

    def __init__(self, n_classes: int, params, label_columns, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(params["MODEL_PATH"], return_dict=True)
        self.hidden_layers = nn.ModuleList()

        last_output = self.bert.config.hidden_size
        if params["EMBEDDING"]=="CLS + MEAN":
            last_output = last_output * 2
            
        
        # Generate Hidden Layers based on params
        if params["HIDDEN_LAYERS"]:
            for (h_layer_size, activation) in params["HIDDEN_LAYERS"]:
                if params["DROPOUT"]:
                    self.hidden_layers.append(nn.Dropout(params['DROPOUT']))
                if activation is not None:
                    self.hidden_layers.append(nn.Linear(last_output, h_layer_size))
                self.hidden_layers.append(activation)
                last_output = h_layer_size

        if params["DROPOUT"] and (params["HIDDEN_LAYERS"] is None):
            self.hidden_layers.append(nn.Dropout(params['DROPOUT']))

        self.classifier = nn.Linear(last_output, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = params["CRITERION"][0]

        self.params = params
        self.label_columns = label_columns

        # self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        
        # Instead of using Pooler directly, we get CLS Token or Mean Pooling etc... 
        if self.params["EMBEDDING"]=="CLS":
            output = self._cls_embeddings(output) # Get CLS Token for Classification
        elif self.params["EMBEDDING"]=="MEAN":
            output = self._meanPooling(output, attention_mask)
        elif self.params["EMBEDDING"]=="CLS + MEAN":
            cls = self._cls_embeddings(output)
            mean = self._meanPooling(output, attention_mask)
            output = torch.cat([cls, mean], dim=1)
        else:
            raise Exception("param[EMBEDDINGS] not valid")
        for layer in self.hidden_layers:
            output = layer(output)
        output_cls = self.classifier(output)

        #output => Sigmoid() => output => BCELoss() => loss
        output = torch.sigmoid(output_cls)

        loss = 0
        if labels is not None:
            if isinstance(self.criterion, nn.BCEWithLogitsLoss): #BCEwithLogitLOss: output => Sigmoid + BCELoss() => loss
                loss = self.criterion(output_cls, labels)
            else:
                loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss, "predictions": outputs, "labels": labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        THRESHOLD_MICRO = max_for_thres(y_pred=predictions, y_true=labels, label_columns=self.label_columns, average="micro")
        THRESHOLD_MACRO = max_for_thres(y_pred=predictions, y_true=labels, label_columns=self.label_columns, average="macro")
        THRESHOLD_CUSTOM = max_for_thres(y_pred=predictions, y_true=labels, label_columns=self.label_columns, average="custom")

        # f1_score = F1Score(task="multilabel", num_labels=len(self.label_columns), threshold=THRESHOLD, average="micro")
        # val_micro_f1 = f1_score(predictions, labels)

        self.log("f1_micro_val_threshold", THRESHOLD_MICRO, logger=True)
        self.log("f1_macro_val_threshold", THRESHOLD_MACRO, logger=True)
        self.log("f1_custom_val_threshold", THRESHOLD_CUSTOM, logger=True)
        # self.log("f1_micro_val", val_micro_f1, prog_bar=True, logger=True)
        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)

        y_pred = predictions.numpy()
        y_true = labels.numpy()

        y_pred = np.where(y_pred > THRESHOLD_CUSTOM, 1, 0)

        class_rep = classification_report(
            y_true,
            y_pred,
            target_names=self.label_columns,
            zero_division=0,
            output_dict=True
        )

        val_macro_f1 = -1
        val_custom_f1 = -1

        for k in class_rep:
            if k == "macro avg":
                val_macro_f1 = class_rep[k]["f1-score"]
                val_macro_recall = class_rep[k]["recall"]
                val_macro_precision = class_rep[k]["precision"]
                if (val_macro_precision + val_macro_recall) != 0:
                    val_custom_f1 = (2*val_macro_recall*val_macro_precision/(val_macro_recall+val_macro_precision))
                else:
                    val_custom_f1 = 0
                self.log(f"custom_f1/Val", val_custom_f1, logger=True)
                self.log(f"{k}_precision/Val", class_rep[k]["precision"], logger=True)
                self.log(f"{k}_recall/Val", class_rep[k]["recall"],  logger=True)
                self.log(f"{k}_f1-score/Val", class_rep[k]["f1-score"], prog_bar=True, logger=True)
                self.log(f"{k}_support/Val", torch.tensor(class_rep[k]["support"], dtype=torch.float32),  logger=True)
            # Avoid that all F1_scores are logged to progress bar..
            else:
                self.log(f"{k}_precision/Val", class_rep[k]["precision"], logger=True)
                self.log(f"{k}_recall/Val", class_rep[k]["recall"],  logger=True)
                self.log(f"{k}_f1-score/Val", class_rep[k]["f1-score"], logger=True)
                self.log(f"{k}_support/Val", torch.tensor(class_rep[k]["support"], dtype=torch.float32),  logger=True)



        for i, name in enumerate(self.label_columns):
            auroc = AUROC(task="binary")
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.log(f"{name}_roc_auc/Val", class_roc_auc, logger=True)

        auroc = AUROC(task="multilabel", num_labels=len(self.label_columns), average="micro")
        total_auroc_micro = auroc(predictions, labels)
        self.log(f"roc_auc_total_micro/Val", total_auroc_micro, prog_bar=True, logger=True)

        auroc = AUROC(task="multilabel", num_labels=len(self.label_columns), average="macro")
        total_auroc_macro = auroc(predictions, labels)
        self.log(f"roc_auc_total_macro/Val", total_auroc_macro, logger=True)

        return {"avg_val_loss":avg_loss, "macro_avg_f1_val":val_macro_f1, "custom_f1_val":val_custom_f1, "roc_auc_total_micro_val":total_auroc_micro}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()


        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.label_columns):
            auroc = AUROC(task="binary")
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.log(f"{name}_roc_auc/Train", class_roc_auc, logger=True)

        auroc = AUROC(task="multilabel", num_labels=len(self.label_columns), average="micro")
        total_auroc = auroc(predictions, labels)
        self.log(f"roc_auc_total_micro/Train", total_auroc, logger=True)

        auroc = AUROC(task="multilabel", num_labels=len(self.label_columns), average="macro")
        total_auroc = auroc(predictions, labels)
        self.log(f"roc_auc_total_macro/Train", total_auroc, logger=True)

        self.log("avg_train_loss", avg_loss, logger=True)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.params['LR'])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def _cls_embeddings(self, output):
        '''Returns the embeddings corresponding to the <CLS> token of each text. '''

        last_hidden_state = output[0]
        cls_embeddings = last_hidden_state[:, 0]
        return cls_embeddings

    def _meanPooling(self, output, attention_mask):
        '''Performs the mean pooling operation. '''

        last_hidden_state = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _maxPooling(self, output, attention_mask):
        '''Performs the max pooling operation. '''

        last_hidden_state = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings
