import numpy as np
import pandas as pd
from keras_preprocessing import sequence

from keras.preprocessing.text import Tokenizer, tokenizer_from_json

# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils.class_weight import compute_class_weight
import ujson as json


class Preprocessing:
    def __init__(self, args):
        self.max_len = args.max_len
        self.max_words = args.max_words
        self.test_size = args.test_size

    def load_data(self):
        self.trainDf = pd.read_csv(
            "/home/alextay96/Desktop/all_workspace/personal_workspace/segment_ocr_payslip/data/train_test/train.csv"
        )
        self.evalDf = pd.read_csv(
            "/home/alextay96/Desktop/all_workspace/personal_workspace/segment_ocr_payslip/data/train_test/eval.csv"
        )
        self.x_train = self.trainDf["text"]
        self.y_train = self.trainDf["label_id"]
        self.x_test = self.evalDf["text"]
        self.y_test = self.evalDf["label_id"]
        self.clsNames = (
            self.evalDf.groupby("label")[["label_id", "label"]]
            .head(1)
            .reset_index()
            .sort_values(by="label_id")
        )["label"].tolist()
        self.clsWeight = compute_class_weight(
            class_weight="balanced", classes=np.unique(self.y_test), y=self.y_test
        )

    def prepare_tokens(self):
        self.tokens = Tokenizer(
            num_words=self.max_words, filters="!#$%&()*+,:;<=>?@[\\]^_`{|}~"
        )
        self.tokens.fit_on_texts(self.x_train)
        with open("./tokenizer.txt", "w") as f:
            f.write(self.tokens.to_json())
        with open("./tokenizer.txt", "r") as f:
            ann = f.read()
        self.tokens = tokenizer_from_json(ann)
        print(self.tokens)

    def sequence_to_token(self, x):
        sequences = self.tokens.texts_to_sequences(x)
        b = sequence.pad_sequences(sequences, maxlen=self.max_len)

        return b
