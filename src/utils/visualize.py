import os
import yaml
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

def visualize_speed_map(map_manager, figsize=(10,10), save_path=None):
    """YAML指定＋origin/resolution考慮でマップ上に速度可視化"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転補正）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=map_manager.waypoints[:, 2], cmap='bwr', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speed [m/s]')
    ax.set_title(f"Speed Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_curve_class_map(map_manager, figsize=(10,10), save_path=None):
    """マップ上にカーブのクラス分類結果を可視化（速度ではなくクラスで色分け）"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # クラス情報
    if not hasattr(map_manager, 'curve_classes'):
        raise AttributeError("MapManagerに 'curve_classes' が存在しません。_compute_speeds で分類が行われているか確認してください。")
    classes = map_manager.curve_classes

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=classes, cmap='tab10', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(classes.min(), classes.max()+1))
    cbar.set_label('Curve Class (0=Straight, ↑Tighter)')
    ax.set_title(f"Curve Class Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_curvature_histogram(map_manager, bins=100, log=False):
    if not hasattr(map_manager, 'waypoints'):
        raise RuntimeError("waypointsがロードされていません。")

    # 曲率取得（再計算してもよいが、ここでは再利用前提）
    wpts = map_manager.waypoints[:, :2]
    N = len(wpts)

    curvature = np.zeros(N, dtype=np.float32)
    for i in range(1, N-1):
        v1 = wpts[i]   - wpts[i-1]
        v2 = wpts[i+1] - wpts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_t = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
        theta = np.arccos(cos_t)
        curvature[i] = theta / n2

    # スムージング
    curvature_smooth = gaussian_filter1d(curvature, sigma=map_manager.smooth_sigma)

    # ヒストグラム描画
    plt.figure(figsize=(10, 5))
    plt.hist(curvature, bins=bins, alpha=0.5, label='Raw', log=log)
    plt.hist(curvature_smooth, bins=bins, alpha=0.7, label='Smoothed', log=log)
    plt.xlabel('Curvature')
    plt.ylabel('Frequency')
    plt.title(f'Curvature Histogram: {map_manager.map_name}')
    plt.legend()
    plt.grid(True)
    plt.show()