FROM python:3.7.13-slim-buster
RUN mkdir /app
WORKDIR /app
COPY . .
RUN apt-get -qq update -y && apt-get -qq install clang g++ libc-dev python3-dev libxrender-dev libxrender1 libsm6 libxext6 swig unzip tzdata gnupg curl wget jq vim -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda init
RUN conda install -c conda-forge -y openbabel
RUN pip install --no-cache-dir \
    tensorflow==2.16.0 \
    streamlit==1.22.0 \
    scikit-learn==1.0.2 \
    pandas==1.3.5 \
    matplotlib==3.5.3 \
    rdkit==2023.3.1 \
    -i https://mirrors.aliyun.com/pypi/simple/
EXPOSE 8501
CMD ["streamlit", "run","app.py","--server.address","0.0.0.0"]
