from pprint import pprint
from typing import List
import pandas as pd
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing import sequence
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import ujson as json
import requests
import glob

with open(
    "./src/app/script_model",
    "rb",
) as f:
    model = torch.jit.load(f)
with open("./src/app/tokenizer.txt", "r") as f:
    ann = f.read()
tokens = tokenizer_from_json(ann)
max_len = 20


def get_cls_name_mapping():
    clsNameMappingCsv = "./src/app/cls_name_mapping.json"

    with open(clsNameMappingCsv, "r") as f:
        mapping = json.load(f)
    return mapping


def process(input: List[str]):
    clsNameMappingCsv = "./app/cls_name_mapping.json"

    with open(clsNameMappingCsv, "r") as f:
        mapping = json.load(f)
    sequences = tokens.texts_to_sequences(input)
    vectorInput = sequence.pad_sequences(sequences, maxlen=max_len)
    inputTensor = torch.from_numpy(vectorInput)
    with autocast():

        output = model(inputTensor)
    predCls = torch.argmax(output, dim=1)
    predNp = predCls.numpy().tolist()
    predClsName = [mapping[str(x)] for x in predNp]
    return predClsName, predNp


def simulate_request():
    clsNameMappingCsv = "./app/cls_name_mapping.json"
    with open(clsNameMappingCsv, "r") as f:
        mapping = json.load(f)
    evalDf = pd.read_csv("./app/eval.csv")
    allClsNameInOrder = [mapping[str(x)] for x in range(12)]
    allGt = np.array([])
    allPred = np.array([])
    for i in tqdm(range(1000)):
        reqBatch = evalDf.sample(n=20)
        reqBatchText = reqBatch["text"].tolist()
        reqGt = reqBatch["label_id"].values
        predClsName, predNp = process(reqBatchText)
        allPred = np.concatenate([allPred, predNp])
        allGt = np.concatenate([allGt, reqGt])
    clsReportJson = classification_report(allGt, allPred, output_dict=True)
    pprint(clsReportJson)
    # metricDf = pd.json_normalize(clsReportJson)


def simulate_request_remote():
    evalDf = pd.read_csv("./src/app/eval.csv")
    allGt = np.array([])
    allPred = np.array([])
    for i in tqdm(range(1000)):
        reqBatch = evalDf.sample(n=20)
        reqBatchText = reqBatch["text"].tolist()
        reqGt = reqBatch["label_id"].values
        resp = requests.post(
            "http://0.0.0.0:8000/predict", json={"input_txt": reqBatchText}
        )
        # predClsName, predNp = process(reqBatchText)
        respBody = resp.json()
        allPred = np.concatenate([allPred, np.array(respBody["prediction_ids"])])
        allGt = np.concatenate([allGt, reqGt])
    clsReportJson = classification_report(allGt, allPred, output_dict=True)
    pprint(clsReportJson)


def sim_upload():
    srcDir = "./data/raw_imgs/**.*"
    allImgs = glob.glob(srcDir, recursive=True)
    for i in tqdm(allImgs):
        resp = requests.post(
            "http://0.0.0.0:8000/files", files={"myFile": open(i, "rb")}
        )
        # predClsName, predNp = process(reqBatchText)
        respBody = resp.json()
        print(respBody)


sim_upload()


def parity_eval():
    evalDf = pd.read_csv("./app/eval.csv")

    with open("./tokenizer.txt", "r") as f:
        ann = f.read()
    tokens = tokenizer_from_json(ann)
    max_len = 20
    with open(
        "/home/alextay96/Desktop/personal_workspace/segment_ocr_payslip/script_model",
        "rb",
    ) as f:
        model = torch.jit.load(f)
    model.eval()
    correct = 0
    with torch.no_grad():
        for _, row in evalDf.iterrows():
            rawStr = str(row["text"])
            label = torch.tensor([row["label_id"]])

            sequences = tokens.texts_to_sequences([rawStr])
            vectorInput = sequence.pad_sequences(sequences, maxlen=max_len)
            inputTensor = torch.from_numpy(vectorInput)
            with autocast():

                output = model(inputTensor)
            predCls = torch.argmax(output, dim=1)
            if predCls[0] == label[0]:
                correct += 1
    acc = correct / len(evalDf)
    print(acc)
