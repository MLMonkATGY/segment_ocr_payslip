from typing import Union

from fastapi import FastAPI
import uvicorn
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
import os
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import torch.nn.functional as F

app = FastAPI()
print(os.getcwd())
with open(
    "./src/app/script_model",
    "rb",
) as f:
    model = torch.jit.load(f)
with open("./src/app/tokenizer.txt", "r") as f:
    ann = f.read()
tokens = tokenizer_from_json(ann)
max_len = 20


class InputReq(BaseModel):
    input_txt: List[str]


def get_cls_name_mapping():
    clsNameMappingCsv = "./src/app/cls_name_mapping.json"

    with open(clsNameMappingCsv, "r") as f:
        mapping = json.load(f)
    return mapping


def process(input: List[str]):
    model.eval()
    clsNameMappingCsv = "./src/app/cls_name_mapping.json"
    with open(clsNameMappingCsv, "r") as f:
        mapping = json.load(f)
    sequences = tokens.texts_to_sequences(input)
    vectorInput = sequence.pad_sequences(sequences, maxlen=max_len)
    inputTensor = torch.from_numpy(vectorInput)

    with torch.no_grad():
        with autocast():

            output = model(inputTensor)
    predCls = torch.argmax(output, dim=1)
    allPredProb = F.softmax(output)
    selectedProb = torch.max(allPredProb, dim=1)
    predNp = predCls.numpy().tolist()
    predClsName = [mapping[str(x)] for x in predNp]
    return predClsName, predNp, selectedProb[0].cpu().numpy()


@app.post("/predict_txt")
def predict(req: InputReq):
    predClsName, predNp = process(req.input_txt)
    return {"predictions": predClsName, "prediction_ids": predNp}


def segments(img: np.ndarray):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 20), np.uint8)
    dilation = cv2.dilate(im_th, kernel, iterations=1)
    output = cv2.connectedComponentsWithStats(dilation, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    allDetectedText = []
    for i in tqdm(
        range(0, numLabels),
        desc="segments",
    ):

        if i == 0:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # (cX, cY) = centroids[i]
        output = img.copy()
        x1 = x
        x2 = x1 + w
        y1 = y
        y2 = y1 + h
        roi = output[y1:y2, x1:x2]
        pil_image = Image.fromarray(roi)
        buf = BytesIO()
        pil_image.save(buf, format="JPEG")

        imgText = pytesseract.image_to_string(pil_image, lang="eng")
        if len(imgText.strip()) > 50 or len(imgText.strip()) < 3:
            continue
        allDetectedText.append(imgText.strip())
        tqdm.write(imgText.strip())
    return allDetectedText


@app.post("/files")
def upload_img(myFile: bytes = File()):
    imageRaw = np.asarray(bytearray(myFile), dtype="uint8")
    img = cv2.imdecode(imageRaw, cv2.IMREAD_COLOR)
    allDetectedText = segments(img)
    predClsName, predNp, selectedProb = process(allDetectedText)
    df = pd.DataFrame(
        {
            "pred_cls": predClsName,
            "pred_ids": predNp,
            "pred_proba": selectedProb,
            "text": allDetectedText,
        }
    )
    df.sort_values(by="pred_proba", ascending=False, inplace=True)
    selectedDf = df.groupby("pred_cls").head(1)
    resp = selectedDf.to_json(orient="records")
    respJson = json.loads(resp)
    return respJson


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
