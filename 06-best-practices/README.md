# Code snippets

## Building and running Docker images

```bash
docker build -t stream-model-duration:v2 .
```

```bash
docker run -it --rm \
    -p 8888:8080 \
    -e PREDICTIONS_STREAM_NAME="ride-predictions" \
    -e RUN_ID="95c848791a7642ff8c26794d43e410a8" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="ap-southeast-1" \
    -v /home/ubuntu/.aws:/root/.aws \
    stream-model-duration:v2
```