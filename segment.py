import cv2
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from typing import List, Dict
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract


def segments(imgPath: str, outputBaseDir: str):

    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 20), np.uint8)
    dilation = cv2.dilate(im_th, kernel, iterations=1)
    fileId = imgPath.split(os.path.sep)[-1].split(".")[0]
    filename = imgPath.split("/")[-1]
    output = cv2.connectedComponentsWithStats(dilation, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in tqdm(
        range(0, numLabels),
        desc="segments",
    ):

        if i == 0:
            continue
        segmentDir = os.path.join(outputBaseDir, fileId).replace(" ", "")
        os.makedirs(segmentDir, exist_ok=True)
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
        filename = "{0}_{1}_{2}_{3}_{4}.png".format(x1, y1, x2, y2, fileId)
        outputFilePath = os.path.join(segmentDir, filename)
        outputFilePath = outputFilePath.replace(" ", "")
        imgText = pytesseract.image_to_string(pil_image, lang="eng")
        if len(imgText) > 50:
            continue
        print(imgText)
        return imgText


imgPath = r"/home/alextay96/Desktop/personal_workspace/segment_ocr_payslip/data/3.webp"
outputBaseDir = (
    r"/home/alextay96/Desktop/personal_workspace/ocr/ocr_segment/segment_output"
)

segments(imgPath, outputBaseDir)
