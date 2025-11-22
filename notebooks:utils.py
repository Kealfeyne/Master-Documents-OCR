import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
from torchvision.ops import box_iou
import numpy as np
from jiwer import wer, cer
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from tqdm import tqdm
# from roboflow import Roboflow


import easyocr
# import pytesseract
# import paddle
# from paddleocr import PaddleOCR
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel


if torch.backends.mps.is_available():
    device = torch.device("mps")

# print(masked_ious(mask_by_bboxes(image, gtbboxes), mask_by_bboxes(image, bboxes)))

def draw_bboxes(image, texts: list, bboxes: list):
    if len(texts) != len(bboxes) or isinstance(texts) == None:
        texts = [""] * len(bboxes)

    for text, xyxy in zip(texts, bboxes):
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text((x1, y1), text, font=ImageFont.truetype("Arial.ttf", size=14))

        image = np.asarray(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def mask_by_bboxes(image, bboxes: list):
    mask = np.zeros(image.shape, dtype=np.int32)

    for xyxy in bboxes:
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        mask[y1:y2, x1:x2] = 1

    return mask

def yolov8_bboxes_to_xyxy(image, bboxes):
    image_size = image.shape[:2]
    bboxes = np.array(bboxes, dtype=np.float32)

    if image_size is not None:
        img_h, img_w = image_size
        bboxes[:, 0] *= img_w  # x_center
        bboxes[:, 1] *= img_h  # y_center
        bboxes[:, 2] *= img_w  # width
        bboxes[:, 3] *= img_h  # height
    
    xyxy_bboxes = np.zeros_like(bboxes)
    xyxy_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1
    xyxy_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1
    xyxy_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x2
    xyxy_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y2
    
    return xyxy_bboxes.astype(np.int32)

# Чтение меток детекции
def read_detection_labels(detection_dataset_path):
    detection_labels = {}

    for item in os.listdir(f"{detection_dataset_path}/images/"):

        image = cv2.imread(f"{detection_dataset_path}/images/{item}")

        file = open(f"{detection_dataset_path}/labels/{item[:-4]}.txt", "r") # .jpg
        data = file.read()
        xywh = np.array([x.split(" ")[1:] for x in data.split("\n")])
        xyxy = yolov8_bboxes_to_xyxy(image, xywh)
        file.close
        
        detection_labels[item] = {"xyxy": xyxy}

    return detection_labels

# Создание eval датасета распознавания
def create_recognition_dataset(detection_dataset_path, detection_labels, recognition_dataset_path):
    for key, value in detection_labels.items():
        image = cv2.imread(f"{detection_dataset_path}/images/{key}")

        i = 0
        for xyxy in value["xyxy"]:
            i += 1
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

            # plt.figure(figsize = (15,15))
            # plt.imshow(image[y1:y2, x1:x2])
            cv2.imwrite(f"{recognition_dataset_path}/unlabeled/{key[:-4]}_{i}.jpg", image[y1:y2, x1:x2])

    return

# Чтение меток распознавания
def read_recognition_labels(recognition_dataset_path):
    recognition_labels = {}

    with open(f"{recognition_dataset_path}/annotations.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
            recognition_labels[data["image"]] = {"text": data["prefix"]}

    return recognition_labels

def masked_ious(image, labels: np.array, preds: np.array):
      SMOOTH = 1e-6

      intersection = (mask_by_bboxes(image, preds) & mask_by_bboxes(image, labels)).sum((0, 1)).max()
      union = (mask_by_bboxes(image, preds) | mask_by_bboxes(image, labels)).sum((0, 1)).max()


      union_by_labels = mask_by_bboxes(image, labels).sum((0, 1)).max()
      union_by_preds = mask_by_bboxes(image, preds).sum((0, 1)).max()

      iou = (intersection + SMOOTH) / (union + SMOOTH)

      iou_by_labels = (intersection + SMOOTH) / (union_by_labels + SMOOTH)
      iou_by_preds = (intersection + SMOOTH) / (union_by_preds + SMOOTH)

      return iou, iou_by_labels, iou_by_preds

def eval_detection(detector, detection_labels, detection_dataset_path):

    ious, ious_by_labels, ious_by_preds = [], [], []

    for key, value in tqdm(detection_labels.items()):
        image = cv2.imread(f"{detection_dataset_path}/images/{key}")
        labels = value["xyxy"]

        preds = detector.detect(image)

        cv2.imwrite(f"evaluation/detection/eval_{key}", draw_bboxes(detector.detection_preprocessing(image), [], preds))

        iou, iou_by_labels, iou_by_preds = masked_ious(image, labels, preds)

        ious.append(iou)
        ious_by_labels.append(iou_by_labels)
        ious_by_preds.append(iou_by_preds)
        
    return ious, ious_by_labels, ious_by_preds

def eval_recognition(recognizer, recognition_labels, recognition_dataset_path):

    cers = []

    for key, value in tqdm(recognition_labels.items()):
        image = cv2.imread(f"{recognition_dataset_path}/{key}")
        labels = value["text"]
        preds = recognizer.recognize(image)

        cers.append(cer(labels, preds))

    return cers




# read_detection_labels("datasets/detection/Documents-OCR-Detection-2/test")
# create_recognition_dataset()
# read_recognition_labels("datasets/recognition/Documents-OCR-Recognition-2/test/annotations.jsonl")


# rf = Roboflow(api_key="4t8iWiXww6199WuWDQXD")
# project = rf.workspace("kealfeyne").project("documents-ocr-detection")
# version = project.version(2)
# dataset = version.download("yolov8")

# rf = Roboflow(api_key="4t8iWiXww6199WuWDQXD")
# project = rf.workspace("kealfeyne").project("documents-ocr-recognition")
# version = project.version(2)
# dataset = version.download("jsonl")
                
                