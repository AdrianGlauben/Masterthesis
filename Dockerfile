FROM nvcr.io/nvidia/pytorch:22.04-py3

MAINTAINER AdrianGlauben

RUN cd /root \
    && git clone https://github.com/AdrianGlauben/Masterthesis --recursive \
    && cd /root/Masterthesis
    && pip install -r requirements.txt
