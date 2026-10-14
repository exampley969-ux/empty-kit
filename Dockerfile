FROM python:3.11-slim
WORKDIR /app
COPY vision_radar/requirements.txt .
# headless OpenCV: no GUI libs needed on a server; httpx only needed for tests
RUN sed -i 's/^opencv-python>=/opencv-python-headless>=/; /^httpx/d' requirements.txt \
 && pip install --no-cache-dir -r requirements.txt
COPY vision_radar/ /app/vision_radar/
ENV VISION_RADAR_DB=:memory: \
    VISION_RADAR_EVIDENCE_DIR=/tmp/evidence \
    VISION_RADAR_MODEL_PATH=/app/vision_radar/models/yolov8n.onnx \
    VISION_RADAR_FRONTEND_DIR=/app/vision_radar/frontend \
    VISION_RADAR_ENABLE_DOCS=0
EXPOSE 10000
# Render injects $PORT (default 10000). ONE worker: sessions live in-process.
CMD ["sh","-c","uvicorn vision_radar.api.fastapi_app:create_app_from_env --factory --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
