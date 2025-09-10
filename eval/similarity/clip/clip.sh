#!/bin/bash
docker run -d --rm --gpus all -p 8002:8002 -v "$(pwd)/cache:/root/.cache" clip