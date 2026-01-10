import os, sys
import numpy as np
import collections
pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(pwd)
sys.path.append(root)
from rl_ppo import PPOAgent

def check_dataset(file_path):
    print(f"正在加载 {file_path} ...")
    try:
        data = np.load(file_path)
        obs = data['obs']
        actions = data['actions']
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # --- 1. 基础形状检查 ---
    print("\n[1. 基础形状检查]")
    print(f"样本总数: {len(actions)}")
    print(f"Obs Shape: {obs.shape} (预期: [N, 45, 4, 9])")
    print(f"Act Shape: {actions.shape} (预期: [N])")
    
    if obs.shape[1:] != (45, 4, 9):
        print("❌ Obs 维度错误！请检查 feature.py 是否为 45 通道。")
    else:
        print("✅ Obs 维度正确。")

    if actions.min() < 0 or actions.max() > 234:
        print(f"❌ Action ID 超出范围 (0-234): min={actions.min()}, max={actions.max()}")
    else:
        print("✅ Action ID 范围合法。")

    # --- 2. 动作分布检查 ---
    print("\n[2. 动作分布统计]")
    # 实例化一个 Agent 仅用于 ID 转换
    dummy_agent = PPOAgent(0)
    
    # 将 ID 转为动作类型 (Play, Chi, Peng, Hu, Pass...)
    act_counts = collections.defaultdict(int)
    for aid in actions:
        resp = dummy_agent.action2response(aid)
        act_type = resp.split()[0] # 取第一个单词
        act_counts[act_type] += 1
        
    total = len(actions)
    for k, v in sorted(act_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<10}: {v:>6} ({v/total:.2%})")

    # 关键逻辑检查
    if 'Chi' not in act_counts:
        print("⚠️ 警告：没有检测到 'Chi' 动作。请检查 parse_action_str 中 last_played_tile 逻辑。")
    if 'Peng' not in act_counts:
        print("⚠️ 警告：没有检测到 'Peng' 动作。请检查 Ignore 解析逻辑。")
    if 'Pass' not in act_counts:
        print("⚠️ 警告：没有检测到 'Pass'。这不正常，说明负样本提取失败。")

    # --- 3. 特征逻辑检查 (手牌数) ---
    print("\n[3. 特征逻辑抽查]")
    # 随机抽查 100 个样本
    indices = np.random.choice(len(actions), 100, replace=False)
    
    hand_cards_14 = 0
    hand_cards_13 = 0
    hand_cards_other = 0
    
    for idx in indices:
        # Channel 2-5 是手牌 (HAND)
        # obs[idx, 2:6, :, :] 是手牌的 4 个 One-hot/Mask 层
        # 简单粗暴求和，看有多少张牌
        # 注意：feature.py 的逻辑是 >=1, >=2... 所以这里直接用 Channel 2 (>=1) 统计牌数即可
        # 或者 sum 所有通道可能会有重叠。
        # 更准确的方法：看 Channel 2 (Hand >= 1) 的 sum
        
        # 修正：check feature.py logic
        # Channel 2: count >= 1
        # Channel 3: count >= 2
        # ...
        # 因此，某张牌的数量 = sum(obs[idx, 2:6, h, w])
        
        # 统计手牌总数
        hand_feature = obs[idx, 2:6, :, :]
        total_cards = np.sum(hand_feature) 
        
        # 由于 feature 编码方式 (>=1, >=2)，sum 刚好等于牌张数
        # 例如 3张五万: >=1(1) + >=2(1) + >=3(1) + ==4(0) = 3
        
        if abs(total_cards - 14) < 0.1:
            hand_cards_14 += 1
        elif abs(total_cards % 3 - 1) < 0.1: # 1, 4, 7, 10, 13
            hand_cards_13 += 1
        else:
            hand_cards_other += 1
            
    print(f"抽查中手牌为 14 张 (摸牌决策) 的样本数: {hand_cards_14}")
    print(f"抽查中手牌为 1/4/7/10/13 张 (响应决策) 的样本数: {hand_cards_13}")
    
    if hand_cards_14 == 0:
        print("⚠️ 警告：没有发现 14 张手牌的样本。说明 Draw 后的 request2obs 可能没把摸到的牌加进去！")
    else:
        print("✅ 检测到 14 张手牌样本，Draw 逻辑正常。")

    # --- 4. 可视化采样 ---
    print("\n[4. 样本可视化翻译]")
    print(f"{'Index':<8} | {'Action String':<20} | {'Hand Count':<10} | {'Visible Count'}")
    print("-" * 60)
    
    # 选取几个典型动作的样本
    sample_indices = []
    # 找一个 Chi
    chi_idx = next((i for i, a in enumerate(actions) if 'Chi' in dummy_agent.action2response(a)), None)
    if chi_idx: sample_indices.append(chi_idx)
    # 找一个 Draw (Play)
    play_idx = next((i for i, a in enumerate(actions) if 'Play' in dummy_agent.action2response(a)), None)
    if play_idx: sample_indices.append(play_idx)
    # 找一个 Pass
    pass_idx = next((i for i, a in enumerate(actions) if dummy_agent.action2response(a) == 'Pass'), None)
    if pass_idx: sample_indices.append(pass_idx)
    
    for idx in sample_indices:
        aid = actions[idx]
        act_str = dummy_agent.action2response(aid)
        
        hand_cnt = int(np.sum(obs[idx, 2:6, :, :]))
        
        # VISIBLE 是 Channel 40-43 (>=1 ... ==4)
        vis_cnt = int(np.sum(obs[idx, 40:44, :, :]))
        
        print(f"{idx:<8} | {act_str:<20} | {hand_cnt:<10} | {vis_cnt}")

if __name__ == "__main__":
    check_dataset("/home/data4T1/lpy/rl-proj/pretrain/human-data/human_data.npz")