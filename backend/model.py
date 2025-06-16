import easyocr
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt

def draw_bboxes(image, texts: list, bboxes: list, bbox_color):
    if bboxes.shape == (1, 0):
        return image
    
    if len(texts) != len(bboxes) or isinstance(texts) == None:
        texts = [""] * len(bboxes)

    for text, xyxy in zip(texts, bboxes):
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text((x1, y1), text, font=ImageFont.truetype("Arial.ttf", size=14))

        image = np.asarray(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

    return image

class OCRModel:
    def __init__(self,):
        self.detector = easyocr.Reader(['ru', 'en'], gpu=False)
        self.recognizer = easyocr.Reader(['ru', 'en'], gpu=False)

        # self.detections = None
        # self.recognitions = None
        # self.groupes = None

        return

    def detection_preprocessing(self, image):
        preprocessed_image = image
        
        return preprocessed_image
    
    def recognition_preprocessing(self, image):
        preprocessed_image = image
        
        return preprocessed_image
    
    def detect(self, image):
        preprocessed_image = self.detection_preprocessing(image)

        detections = np.array(self.detector.detect(preprocessed_image)[0][0])
        if detections.shape == (0,):
            detections = np.array([[]])
        else:
            detections[:,1], detections[:,2] = detections[:,2].copy(), detections[:,1].copy()

        # self.detections = detections
        return detections
        
    
    def recognize(self, image):
        preprocessed_image = self.recognition_preprocessing(image)
        recognitions = self.recognizer.recognize(preprocessed_image)
        recognitions = [x[1] for x in recognitions]

        # self.recognitions = recognitions
        return recognitions
    
    def group_texts(self, bboxes, texts, threshold=30):
        lines = []
        current_line = []
        
        for i, (word, box) in enumerate(zip(texts, bboxes)):
            if not current_line:
                current_line.append((word, box))
            else:
                last_box = current_line[-1][1]
                if abs(box[1] - last_box[1]) < threshold:  # Проверка по Y-координате
                    current_line.append((word, box))
                else:
                    lines.append(current_line)
                    current_line = [(word, box)]
        if current_line:
            lines.append(current_line)

        groups = []
        for line in lines:
            # print(" ".join([x[0] for x in group]))
            group = " ".join([" ".join([i for i in x[0]]) for x in line])
            groups.append(group)

        # self.groups = groups
        return groups
    
    def forward(self, image):
        detections = self.detect(image)

        recognitions = []

        for detection in detections:
            cropped_image = image[detection[1]:detection[3], detection[0]:detection[2]]
            recognition = self.recognize(cropped_image)
            recognitions.append(recognition)
            # plt.imshow()

        groups = self.group_texts(detections, recognitions)

        return groups