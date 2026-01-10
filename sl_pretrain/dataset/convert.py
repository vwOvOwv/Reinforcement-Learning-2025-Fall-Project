import os
import ast
import uuid
from tqdm import tqdm
import argparse

def get_chi_middle_tile(tiles):
    """
    ['T2', 'T3', 'T4'] -> T3
    """
    def sort_key(t):
        return (t[0], int(t[1]))
    
    valid_tiles = [t for t in tiles if len(t) >= 2]
    if len(valid_tiles) < 3: return valid_tiles[0] if valid_tiles else "XX"
    
    sorted_tiles = sorted(valid_tiles, key=sort_key)
    return sorted_tiles[1]

def convert_file(content, path):
    lines = content.splitlines()
    initial_hands = {}
    acts = []
    has_played = {str(i): False for i in range(4)}
    output_lines = []

    # generate match id
    output_lines.append(f"Match {uuid.uuid4()}, Source: {path}")
    
    wind_map = {'东': 0, '南': 1, '西': 2, '北': 3}
    act_map = {
        '摸牌': 'Draw', 
        '补花后摸牌': 'Draw', # ignore Hua
        '杠后摸牌': 'Draw',
        '打牌': 'Play', 
        '碰': 'Peng', 
        '吃': 'Chi',
        '明杠': 'Gang', 
        '暗杠': 'AnGang', 
        '补杠': 'BuGang',
        '和牌': 'Hu', 
        '自摸': 'Hu', 
        '抢杠和': 'Hu'
    }

    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        
        if parts[0] in wind_map:
            output_lines.append(f"Wind {wind_map[parts[0]]}")
            continue

        if parts[0].isdigit() and len(parts) > 1 and parts[1].startswith('['):
            seat = parts[0]
            tiles = ast.literal_eval(parts[1])
            # filter out Hua tiles
            clean_tiles = [t for t in tiles if not t.startswith('H')]
            initial_hands[seat] = clean_tiles
            continue

        if parts[0].isdigit() and len(parts) > 1:
            seat = parts[0]
            raw_act = parts[1]
            
            # ignore 补花
            if raw_act == '补花': 
                continue
            
            std_act = act_map.get(raw_act)
            if not std_act: 
                continue
            tile_info = ""
            list_start = line.find('[')
            list_end = line.find(']')
            
            if list_start != -1 and list_end != -1:
                tiles_list = ast.literal_eval(line[list_start:list_end+1])
                if all(t.startswith('H') for t in tiles_list):  # All tiles are Hua
                    continue

                if std_act == 'Chi':
                    tile_info = get_chi_middle_tile(tiles_list)
                else:
                    tile_info = tiles_list[0]

            if raw_act == '补花后摸牌' and len(initial_hands.get(seat, [])) < 13:
                if seat in initial_hands:
                    initial_hands[seat].append(tile_info)
                continue

            if raw_act != '补花后摸牌':
                has_played[seat] = True

            acts.append(f"Player {seat} {std_act} {tile_info}")

    deferred_draws = []
    for s in sorted(initial_hands.keys()):
        hand = initial_hands[s]
        if len(hand) == 14:
            drawn = hand.pop()
            deferred_draws.append(f"Player {s} Draw {drawn}")
        output_lines.append(f"Player {s} Deal {' '.join(hand)}")
    
    output_lines.extend(deferred_draws)
    output_lines.extend(acts)
    output_lines.append("Score 0 0 0 0")
    return "\n".join(output_lines) + "\n\n"


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--load_path', type=str, help="Path to raw data dir",
                            default='/home/data4T1/lpy/rl-proj/pretrain/human-data/output2017/')
    arg_parser.add_argument('--save_path', type=str, help="Path to save processed data",
                            default='/home/data4T1/lpy/rl-proj/pretrain/human-data/output2017/merged.txt')
    args = arg_parser.parse_args()
    data_path = args.load_path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    
    file_paths = []
    print(f"Scanning {data_path} ...")
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt"):
                file_paths.append(os.path.join(root, file))
    
    print(f"Total matches found: {len(file_paths)}")
    print(f"Start converting and merging into {args.save_path} ...")

    with open(args.save_path, 'w', encoding='utf-8') as f_out:
        for path in tqdm(file_paths):
            # print(path)
            with open(path, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
                standard_log = convert_file(content, path)
            f_out.write(standard_log)