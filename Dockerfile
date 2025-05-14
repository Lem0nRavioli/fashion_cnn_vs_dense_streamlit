FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow

# copy only model folder for now
COPY model/ ./model/

EXPOSE 5000

# set default training script
ENV TRAIN_SCRIPT=train_dense.py

CMD python model/$TRAIN_SCRIPT && mlflow ui --host 0.0.0.0 --port 5000
