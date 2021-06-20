FROM python:3.8

RUN mkdir /project
RUN mkdir -p /project/script/
RUN mkdir -p /project/data/


COPY requirements.txt /project/
COPY umap_hdbscan/* project/script/
RUN pip install -r /project/requirements.txt

WORKDIR /project/script/

CMD ["python", "gridsearch.py"]
