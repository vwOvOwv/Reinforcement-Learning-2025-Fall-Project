import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(pwd))
sys.path.append(root)
from rl_ppo import PPOAgent
import argparse
import numpy as np
import tqdm

def parse_action_str(action, tile, played_tile=None):
    if action == 'Play': return f"Play {tile}"
    elif action == 'Hu': return "Hu"
    elif action == 'AnGang': return f"AnGang {tile}"
    elif action == 'BuGang': return f"BuGang {tile}"
    elif action == 'Peng': return f"Peng {tile}"
    elif action == 'Gang': return f"Gang {tile}"
    elif action == 'Chi':
        if played_tile is None: 
            return None
        return f"Chi {tile} {played_tile}"
    return None

def process_match(full_log):
    agents = [PPOAgent(i) for i in range(4)]
    obs_list = []
    act_list = []
    for idx, request in enumerate(full_log):
        request = request.strip()
        tokens = request.split()
        if not tokens or tokens[0] in ['Match', 'Score', 'Fan', 'Huang']:
            continue
        # print(request)
        if tokens[0] == 'Player' and tokens[2] == 'Play':
            last_played_tile = tokens[3]
        
        # find agents that should take action (those whose request2obs return not None)
        active_agents = []  # (agent_idx, observation), agents that take action
        for seat in range(4):
            tmp_request = request
            if tokens[0] == 'Player' and tokens[2] == 'Deal':
                if int(tokens[1]) == seat:
                    tmp_request = "Deal " + " ".join(tokens[3:])
                    ret = agents[seat].request2obs(tmp_request)
                    if ret is not None: 
                        active_agents.append((seat, ret['observation']))
            elif tokens[0] == 'Player' and tokens[2] == 'Draw':
                if int(tokens[1]) == seat:
                    tmp_request = "Draw " + tokens[3]
                    ret = agents[seat].request2obs(tmp_request)
                else:
                    ret = agents[seat].request2obs(tmp_request)
                if ret is not None:
                    active_agents.append((seat, ret['observation']))
            else:
                ret = agents[seat].request2obs(tmp_request)
                if ret is not None:
                    active_agents.append((seat, ret['observation']))

        if not active_agents:
            continue
        if idx + 1 >= len(full_log): 
            break
        
        # find which agent takes action and what action
        next_seat = -1
        next_response = None
        next_line = full_log[idx + 1].strip()
        next_tokens = next_line.split()
        if len(next_tokens) >= 3 and next_tokens[0] == 'Player':
            seat = int(next_tokens[1])
            act = next_tokens[2]  # Draw, Play, Hu, Chi, Peng, Gang...
            tile = next_tokens[3] if len(next_tokens) > 3 else None
            if act != 'Draw':
                next_seat = seat
                played_tile = last_played_tile if act == 'Chi' else None  # type: ignore
                next_response = parse_action_str(act, tile, played_tile)
        
        ignore_map = {} # seat -> action_str
        if 'Ignore' in next_line:
            ignored_responses = next_line.split('Ignore')
            for response in ignored_responses[1:]:
                response_tokens = response.strip().split()
                if len(response_tokens) >= 3 and response_tokens[0] == 'Player':
                    seat = int(response_tokens[1])
                    act = response_tokens[2]
                    tile = response_tokens[3] if len(response_tokens) > 3 else None
                    played_tile = last_played_tile if act == 'Chi' else None  # type: ignore
                    response = parse_action_str(act, tile, played_tile)
                    if response:
                        ignore_map[seat] = response
        
        for seat, obs in active_agents:
            target_act = agents[seat].OFFSET_ACT['Pass']
            if seat == next_seat and next_response: # agent taking action
                target_act = agents[seat].response2action(next_response)
            elif seat in ignore_map:    # agents taking action but ignored
                target_act = agents[seat].response2action(ignore_map[seat])
            else:   # agents not taking action (Pass by default)
                if len(agents[seat].valid) <= 1:    # no other choices
                    continue
            if tokens[0] == 'Player' and tokens[2] == 'Draw' and target_act == agents[seat].OFFSET_ACT['Pass']:
                raise ValueError(f"Draw action should not be Pass.")
            obs_list.append(obs)
            act_list.append(target_act)

    return obs_list, act_list
                
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--load_path', type=str, help="Path to raw data",
                            default='/home/data4T1/lpy/rl-proj/pretrain/human-data/output2017/merged.txt')
    arg_parser.add_argument('--save_path', type=str, help="Path to save processed data",
                            default='/home/data4T1/lpy/rl-proj/pretrain/human-data/output2017/output2017.npz')
    args = arg_parser.parse_args()
    data_path = args.load_path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_name, ext = os.path.splitext(os.path.basename(args.save_path))

    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    raw_matches = content.split('Match ')
    all_obs = []
    all_acts = []
    CHUNK_SIZE = 2000000
    part_idx = 0        
    print(f"Total matches found: {len(raw_matches) - 1}")

    for match_idx, match_str in enumerate(tqdm.tqdm(raw_matches)):
        if not match_str.strip():
            continue

        full_log = ("Match " + match_str).strip().split('\n')
        try:
            obs, acts = process_match(full_log)
        except Exception:
            print(f"Skipping match {full_log[0]} due to data corruption caused by Hua tiles.")
            continue
        all_obs.extend(obs)
        all_acts.extend(acts)

        if len(all_obs) >= CHUNK_SIZE:
            save_obs = np.array(all_obs[:CHUNK_SIZE], dtype=np.float32)
            save_acts = np.array(all_acts[:CHUNK_SIZE], dtype=np.int64)
            
            current_save_path = os.path.join(save_dir, f"{base_name}_{part_idx}{ext}")
            
            print(f"Saving part {part_idx} to {current_save_path} (Size: {CHUNK_SIZE})...")
            np.savez_compressed(current_save_path, obs=save_obs, actions=save_acts)
            
            all_obs = all_obs[CHUNK_SIZE:]
            all_acts = all_acts[CHUNK_SIZE:]
            
            part_idx += 1

        # break  # Remove this break to process all matches

    if len(all_obs) > 0:
        save_obs = np.array(all_obs, dtype=np.float32)
        save_acts = np.array(all_acts, dtype=np.int64)
        current_save_path = os.path.join(save_dir, f"{base_name}_{part_idx}{ext}")
        print(f"Saving final part {part_idx} to {current_save_path} (Size: {len(all_obs)})...")
        np.savez_compressed(current_save_path, obs=save_obs, actions=save_acts)