# Mahjong Bot with Supervised Pretraining and Distributed PPO

This repo contains implementation of supervised pretraining and distributed PPO applied to Chinese Standard Mahjong.

## Environment Setup
Clone this repo:

```bash
git clone git@github.com:vwOvOwv/Reinforcement-Learning-2025-Fall-Project.git
```

Build dependencies:

```bash
conda create -n mahjong-rl python=3.10 -y
conda activate mahjong-rl
pip install -r requirements.txt
```

Note that you may need to manually install [MahjongGB](https://github.com/ailab-pku/PyMahjongGB).

## Supervised Pretraining

### Data Preparation

We use ~380k human game logs from Botzone to train the backbone and policy head. The logs can be downloaded from:

[Botzone Log1 (EN)](https://disk.pku.edu.cn/anyshare/en-us/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA?_tb=none&expires_at=2026-04-30T23%3A59%3A48%2B08%3A00&item_type=&password_required=false&title=data.zip&type=anonymous)

[Botzone Log2 (CN) password: rm79](https://pan.baidu.com/s/1vXzYUsRBNpH245SQku0b3A#list/path=%2F)

Then preprocess the raw data using the provided scripts in the `sl_pretrain/dataset`.

First, convert CN logs to EN logs:

```bash
python sl_pretrain/dataset/convert.py --load_path PATH_TO_CN_LOGS --save_path PATH_TO_CONVERTED_EN_LOGS
```

Then, preprocess the logs into `(observation, action)` pairs, which will be saved as `.npz` files:

```bash
python sl_pretrain/dataset/preprocess.py --load_path PATH_TO_EN_LOGS --save_path PATH_TO_PROCESSED_DATA
```

We provide the processed data at [Processed Data Link]().

### Pretraining

To train the supervised learning model, run:

```bash
python sl_pretrain.py --config configs/sl_pretrain.yaml
```

You may need to adjust the paths in the config file accordingly. It takes ~1.5 hours to train 1 epoch on a single NVIDIA RTX 4090 GPU.

We also provide weights of the pretrained model (ResNet-34 backbone) at [Pretrained Model Link]().

## PPO Training

After pretraining, fine-tune the model using distributed PPO.

```bash
python rl_train.py --config configs/rl_train.yaml
```

You may need to adjust the paths in the config file accordingly.

## Submission

To be consistent with botzone I/O format, revise the first line of `rl_ppo/agents/ppo_agent.py`
before packing.

```python
from base_agent import MahjongGBAgent   # .base_agent => base_agent
```

Then run the shell script.

```bash
./pack.sh PATH_TO_MODEL_WEIGHTS
```

Download the output file `handin.zip` from server and decompress it locally. Its
orgainization should be like:

```
handin/
  ├── codes.zip
  └── testrl.pt
```

Finally, submit `codes.zip` and `testrl.pt` to Botzone to test your bot :)