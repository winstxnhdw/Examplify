FROM ghcr.io/theroyallab/tabbyapi@sha256:99a0b1804fa0d700230fbac157b7ff120a94cbc7be0fb408c39ec325d1fa89a6

WORKDIR /app

RUN python3 /app/main.py download turboderp/Qwen3.8-27B-exl3 \
    --revision 4acd9ad5af224ec9e8815a54d71a033f378a7e9d \
    --folder-name Qwen3.8-27B-exl3-SC_4.00bpw_H5

ENV HF_HUB_OFFLINE=1
