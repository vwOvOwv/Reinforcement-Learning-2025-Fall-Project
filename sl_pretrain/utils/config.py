import yaml

def config_parser(path):
    required = [
        'data_path', 'output_dir',
        'epochs', 'batch_size', 'learning_rate',
        'optimizer', 'momentum', 'weight_decay',
        'scheduler',
        'loss',
    ]
    defaults = {
        'num_workers': 8,
        'dropout': 0,
        'zero_init_residual': False,
        'log_interval': 50
    }
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
            for key in config.keys():
                if key not in required:
                    raise KeyError(f"{key} is not included in {path}")
            for key in defaults.keys():
                if key not in config:
                    config[key] = defaults[key]
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{path}' was not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")