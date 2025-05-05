import numpy as np
import torch
import torch.nn as nn
import numpy as np

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    ターゲットネットワークのパラメータをソフト更新
    """
    for target_param, src_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + src_param.data * tau
        )

## actionを変換
def convert_action(action, steer_range: float=0.5, speed_range: float=8.0):
    
    steer = action[0] * steer_range
    speed = (action[1] + 1.0) * 0.5 * speed_range

    
    return [steer, speed]

def convert_scan(scan, max_range: float=20.0):
    scan = np.clip(scan, 0, max_range)
    scan = scan / max_range
    return scan

