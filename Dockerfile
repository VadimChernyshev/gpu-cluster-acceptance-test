FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /workspace

COPY train_ddp.py .
COPY test_smoke.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train_ddp.py"]
CMD []
