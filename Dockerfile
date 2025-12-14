FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

RUN mkdir -p /app/data
RUN chmod +x /app/run.sh || true

CMD ["bash", "/app/run.sh"]