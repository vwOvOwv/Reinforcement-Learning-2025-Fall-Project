#!/bin/bash

CKPT_PATH=$1

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: ./pack.sh <path_to_checkpoint.pt>"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: File '$CKPT_PATH' does not exist"
    exit 1
fi

mkdir handin
cp __main__.py handin
cp rl_ppo/models/network.py handin
cp rl_ppo/agents/ppo_agent.py handin
cp rl_ppo/agents/base_agent.py handin

cd handin
zip -r "codes.zip" __main__.py network.py ppo_agent.py base_agent.py
cp $CKPT_PATH testrl.pt
zip -r "../handin.zip" codes.zip testrl.pt
cd ..

rm -r handin