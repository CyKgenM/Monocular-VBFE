# syntax=docker/dockerfile:1

FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

COPY requirements.txt /home/jovyan/

RUN pip install --no-cache-dir -r requirements.txt

COPY /notebooks/ /home/jovyan/
COPY /include/ /home/jovyan/
COPY main.py extract.py inference.py /home/jovyan/