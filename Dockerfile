FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG package_version

WORKDIR /app

COPY dist dist

RUN pip3 install --find-links=file:dist perceiver-io[image,text]==$package_version
