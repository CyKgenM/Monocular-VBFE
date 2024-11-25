# syntax=docker/dockerfile:1

FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

COPY requirements.txt /home/jovyan/

RUN pip install --no-cache-dir -r requirements.txt

COPY main.ipynb mono.ipynb pointnet2_utils.py pointnet2_ssg.py /home/jovyan/
COPY /data/ /home/jovyan/