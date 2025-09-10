#!/bin/bash

cd ~/ulip2_encoder
docker run -d --rm --gpus all -p 8003:8003 -v "$(pwd)/weights/ulip_models:/app/ULIP/ulip_models" ulip2_server
cd ~/ip-scenecraft