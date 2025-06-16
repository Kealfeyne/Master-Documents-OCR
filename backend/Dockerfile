ARG DOCKER_BACKEND_PREIMAGE_NAME

FROM $DOCKER_BACKEND_PREIMAGE_NAME:latest

WORKDIR container/

COPY . .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

CMD python api.py