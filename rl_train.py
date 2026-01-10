import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_ppo import Actor, Learner, ReplayBuffer
import time
import argparse
import random
import numpy as np
import torch
import yaml

def set_random_state(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', 
                        help='Path of config file', required=True)
    parser.add_argument('--seed', type=int, default=2025, 
                        help='Random seed (default: 2025)')
    run_id = time.strftime("%Y%m%d_%H%M%S")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    if args.config is None:
        raise Exception('Unrecognized config file: {args.config}.')
    else:
        config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config['global']['run_id'] = run_id
    config['learner']['run_id'] = run_id
    config['actor']['run_id'] = run_id

    replay_buffer = ReplayBuffer(config['global']['replay_buffer_size'], 
                                 config['global']['replay_buffer_episode'])
    
    # create actors and learner, each actor runs in its own process
    actors = []
    for i in range(config['actor']['num_actors']):
        actor = Actor(config['actor'], i, replay_buffer)
        actors.append(actor)

    learner = Learner(config['learner'], replay_buffer)
    
    for actor in actors: 
        actor.start()
    learner.start()
    
    for actor in actors: 
        actor.join()
    learner.terminate()

    print("Training finished.")