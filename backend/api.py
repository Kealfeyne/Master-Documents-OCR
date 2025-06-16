import os
from datetime import datetime
import io
import aiofiles
import base64

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
import numpy as np

from model import OCRModel, draw_bboxes


TMP_DIR = f'tmp_files/'
IMAGE_NAME = f'image.jpg'
TEXT_NAME = f'text.txt'

app = FastAPI()

model = OCRModel()


@app.get("/")
def hello():
    return "Все робит на тесте"

async def save_image(file_binary, filename=None):
    if filename is None:
        # filename = f'tmp_{datetime.now()}_.jpg'
        filename = IMAGE_NAME

    img_file_path = os.path.join(TMP_DIR, filename)

    async with aiofiles.open(img_file_path, 'wb') as out_file:
        await out_file.write(file_binary) 

    return img_file_path, file_binary

async def save_text(text, filename=None):
    if filename is None:
        # filename = f'tmp_{datetime.now()}_.jpg'
        filename = TEXT_NAME

    text_file_path = os.path.join(TMP_DIR, filename)

    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    
    return text_file_path

@app.post("/detect/")
async def detect(file: bytes = File(...)):
    print("Зашли в detect")
    if not file:
        return {"message": "No upload file sent"}
    else:
    
        img_file_path, binary_img_data = await save_image(file)

        # image = cv2.cvtColor(cv2.imread(f"datasets/manual/shkolin dl 1.jpg"), cv2.COLOR_BGR2RGB)
        image = Image.open(img_file_path).convert("RGB")
        output = "\n".join(model.forward(np.array(image)))

        print(output)
        text_file_path = await save_text(output)

        # predict_img = Image.open(img_file_path) #predict_img_path
        with open(img_file_path, "rb") as f:
            image_file = f.read()
        # bytes_image = io.BytesIO()

        content = {
            # "img_file_path": img_file_path,
            "bimage": base64.b64encode(image_file),
            "text": output.encode("utf-8")
        }

        # predict_img.save(bytes_image, format='PNG')
        
        # return Response(content=content) # , headers=response, media_type="image/png"
        return content
    
@app.post("/refresh/")
async def refresh():
    # image = Image.open(os.path.join(TMP_DIR, IMAGE_NAME)).convert("RGB")
    # output = 
    try:
        with open(os.path.join(TMP_DIR, IMAGE_NAME), "rb") as f:
                image_file = f.read()

        with open(os.path.join(TMP_DIR, TEXT_NAME), "r", encoding="utf-8") as f:
                text = f.read()
    except:
        image_file = None

    if image_file:
        content = {
            "bimage": base64.b64encode(image_file),
            "text": text
        }
    else:
        content = {}

    print(content)

    return content

@app.post("/delete/")
async def delete():

    try:
        if os.path.exists(os.path.join(TMP_DIR, IMAGE_NAME)):
            os.remove(os.path.join(TMP_DIR, IMAGE_NAME))

        if os.path.exists(os.path.join(TMP_DIR, TEXT_NAME)):
            os.remove(os.path.join(TMP_DIR, TEXT_NAME))
    except:
        return


if __name__ == '__main__':
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACKEND_PORT'))
