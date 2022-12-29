import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from src.txt_cls.utils import Preprocessing
from src.txt_cls.model import TweetClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.txt_cls.parser import parameter_parser
from torchmetrics import Accuracy
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
import pandas as pd
import ujson as json
from sklearn.metrics import classification_report


class DatasetMaper(Dataset):
    """
    Handles batches of dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx])
        return self.x[idx], self.y[idx]


class Execute:
    """
    Class for execution. Initializes the preprocessing as well as the
    Tweet Classifier model
    """

    def __init__(self, args):
        self.__init_data__(args)
        # self.device = torch.device("cuda")
        self.args = args
        self.batch_size = args.batch_size
        self.device = torch.device("cpu")
        self.model = TweetClassifier(args)
        self.outputRunDir = "/home/alextay96/Desktop/all_workspace/personal_workspace/segment_ocr_payslip/output"
        os.makedirs(self.outputRunDir, exist_ok=True)
        self.criterion = CrossEntropyLoss(
            weight=torch.tensor(self.preprocessing.clsWeight, dtype=torch.float32).to(
                self.device
            )
        )

    def __init_data__(self, args):
        """
        Initialize preprocessing from raw dataset to dataset split into training and testing
        Training and test datasets are index strings that refer to tokens
        """
        self.preprocessing = Preprocessing(args)
        self.preprocessing.load_data()
        self.preprocessing.prepare_tokens()
        self.clsName = self.preprocessing.clsNames
        raw_x_train = self.preprocessing.x_train
        raw_x_test = self.preprocessing.x_test

        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test
        print(len(self.y_train))
        print(len(self.y_test))

        self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
        self.x_test = self.preprocessing.sequence_to_token(raw_x_test)

    def train(self):
        num_classes = len(self.y_test.unique())
        training_set = DatasetMaper(self.x_train, self.y_train)
        test_set = DatasetMaper(self.x_test, self.y_test)

        self.loader_training = DataLoader(training_set, batch_size=200, shuffle=True)
        self.loader_test = DataLoader(test_set)

        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        allAcc = []
        self.model = self.model.to(self.device)
        for epoch in range(args.epochs):
            t_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
                self.device
            )
            e_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            predictions = []
            trainGt = []
            self.model.train()
            self.model.to(self.device)

            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)
                x = x.to(self.device)
                y = y.to(self.device)
                with autocast():
                    y_pred = self.model(x)
                    # y = y.view(-1, 1)
                    loss = self.criterion(y_pred, y)
                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                predictions += list(y_pred.squeeze().detach())
                trainGt += y
            test_predictions, test_gt = self.evaluation()

            trainGt = torch.stack(trainGt)
            predictions = torch.stack(predictions)
            train_accuary = t_accuracy(predictions, trainGt)
            test_accuracy = e_acc(test_predictions, test_gt)
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                test_gt_np = test_gt.detach().cpu().numpy()
                preds = test_predictions.argmax(1)
                test_predictions_np = preds.detach().cpu().numpy()

                ConfusionMatrixDisplay.from_predictions(
                    test_gt_np,
                    test_predictions_np,
                    normalize="true",
                    display_labels=list(range(num_classes)),
                )
                plt.savefig(
                    "{0}/balance_norm_e{1}_acc_{2}.pdf".format(
                        self.outputRunDir, epoch, test_accuracy
                    )
                )
                plt.close("all")
                ConfusionMatrixDisplay.from_predictions(
                    test_gt_np,
                    test_predictions_np,
                    # normalize="false",
                    display_labels=list(range(num_classes)),
                )
                plt.savefig(
                    f"{self.outputRunDir}/balance_abs_e{epoch}_acc{test_accuracy}.pdf"
                )
                plt.close("all")
                if epoch == args.epochs - 1:
                    metricOutput = classification_report(
                        test_gt_np,
                        test_predictions_np,
                        target_names=self.preprocessing.clsNames,
                        output_dict=True,
                    )
                    cm = confusion_matrix(test_gt_np, test_predictions_np)
                    allAccByPart = cm.diagonal()
                    metricOutput["train_loss"] = loss.item()
                    metricOutput["train_avg_acc"] = float(train_accuary.numpy())

                    for i, (k, v) in enumerate(
                        zip(self.preprocessing.clsNames, allAccByPart)
                    ):
                        metricOutput[k]["accuracy"] = float(v)
                        metricOutput[k]["label_id"] = i

                    print(metricOutput)
                    with open("./all_metrics.json", "w") as f:
                        json.dump(metricOutput, f)

            print(
                "Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f"
                % (epoch + 1, loss.item(), train_accuary, test_accuracy)
            )

    def evaluation(self):

        predictions = []
        evalGt = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)
                x = x.to(self.device)
                y = y.to(self.device)
                with autocast():
                    y_pred = self.model(x)
                predictions += list(y_pred.detach())
                evalGt += y
        return torch.stack(predictions), torch.stack(evalGt)


if __name__ == "__main__":

    args = parameter_parser()

    execute = Execute(args)
    execute.train()
    model = execute.model.to(torch.device("cpu"))
    scriptModel = torch.jit.script(execute.model)
    with open("./script_model", "wb") as f:
        torch.jit.save(scriptModel, f)
    evalDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/personal_workspace/segment_ocr_payslip/data/train_test/eval.csv"
    )
    clsNameRaw = (
        evalDf[["label", "label_id"]].groupby("label").head(1).reset_index()
    ).to_json(orient="records")
    b = json.loads(clsNameRaw)
    mapping = {int(x["label_id"]): x["label"] for x in b}
    with open("./cls_name_mapping.json", "w") as f:
        json.dump(mapping, f)
