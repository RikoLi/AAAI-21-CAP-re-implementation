from fastreid.config.config import CfgNode

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = CfgNode.load_cfg(f)
    return cfg