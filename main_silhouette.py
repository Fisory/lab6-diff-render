"""
实验六 Part 1：基于可微剪影的形状优化
将球体网格通过梯度下降逐步优化为奶牛形状。

核心思路：
  - 用软光栅化（SoftSilhouetteShader）让边界处的梯度非零
  - 从 20 个视角同时计算剪影误差，避免单视角的歧义性
  - 加入拉普拉斯 / 边长 / 法线三项正则化，防止网格拓扑崩坏
"""

import os
import warnings
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import imageio
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
import pytorch3d.loss as mesh_loss

from utils import (
    get_device,
    get_cow_mesh_path,
    normalize_mesh,
    setup_cameras,
    build_silhouette_renderer,
)

# ─────────────────────────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── 渲染质量（CPU 友好：减小这三项；GPU 可调回 num_views=20, image_size=128, fpp=50）
    "num_views":       8,       # 参与优化的相机视角数（CPU 上建议 8，GPU 可用 20）
    "image_size":      64,      # 渲染分辨率（CPU 上 64，GPU 可用 128）
    "faces_per_pixel": 20,      # 每像素最多覆盖的三角面数（CPU 上 20，GPU 可用 50）
    # ── 网格细分
    "sphere_level":    4,       # ico_sphere 细分等级：4 → 2562 顶点 / 5120 面
    # ── 优化超参
    "num_iter":        300,     # 优化迭代总步数
    "lr":              1e-2,    # Adam 初始学习率
    "sigma":           1e-4,    # 软光栅化模糊参数 σ
    # ── 正则化权重
    "w_lap":           0.1,     # 拉普拉斯平滑
    "w_edge":          1.0,     # 边长一致性
    "w_normal":        0.01,    # 法线一致性
    # ── 输出
    "save_dir":        "output/silhouette",
    "save_interval":   30,      # 每隔多少步保存一帧
    "gif_fps":         5,
    "vis_views":       [0, 2, 4, 6],  # 可视化视角索引（从 8 个里选 4 个）
}


# ─────────────────────────────────────────────────────────────────────────────
# 渲染工具
# ─────────────────────────────────────────────────────────────────────────────

def render_silhouettes(mesh, cameras, renderer):
    """
    把 mesh 扩展到 N 份（对应 N 个视角），批量渲染后取 alpha 通道。
    返回形状：(N, H, W)
    """
    n = len(cameras)
    images = renderer(mesh.extend(n), cameras=cameras)
    return images[..., 3]


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_frame(
    step, num_iter, target_sil, pred_sil, loss_total, loss_sil, save_dir,
    vis_views=None,
):
    """保存多视角 Ground Truth vs Optimizing 对比图。"""
    if vis_views is None:
        vis_views = [0]
    n = len(vis_views)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, None]
    fig.suptitle(
        f"Step {step}/{num_iter}  |  Total Loss: {loss_total:.4f}  |  Sil Loss: {loss_sil:.4f}",
        fontsize=12,
    )
    for col, vi in enumerate(vis_views):
        axes[0, col].imshow(target_sil[vi].detach().cpu().numpy(), cmap="gray")
        axes[0, col].set_title(f"Target  (view {vi})", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(pred_sil[vi].detach().cpu().numpy(), cmap="gray")
        axes[1, col].set_title(f"Predicted (view {vi})", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout()
    path = os.path.join(save_dir, f"frame_{step:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def save_loss_curve(loss_log, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"total": "black", "sil": "steelblue", "lap": "orange",
              "edge": "green", "normal": "red"}
    for key, vals in loss_log.items():
        ax.plot(vals, label=key, color=colors.get(key))
    ax.set_xlabel("迭代步数")
    ax.set_ylabel("损失值")
    ax.set_title("优化过程损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = get_device()

    # 1. 载入并归一化目标网格 ────────────────────────────────────────────────
    print("\n[1/5] 加载目标奶牛网格...")
    obj_path = get_cow_mesh_path()
    target_mesh = load_objs_as_meshes([obj_path], device=device)
    target_mesh = normalize_mesh(target_mesh)

    # 2. 构建相机和渲染器，渲染目标剪影 ─────────────────────────────────────
    print("[2/5] 渲染目标剪影（共 {} 视角）...".format(CFG["num_views"]))
    cameras = setup_cameras(CFG["num_views"], device=device)
    renderer = build_silhouette_renderer(
        cameras, CFG["image_size"], CFG["sigma"], CFG["faces_per_pixel"]
    )

    with torch.no_grad():
        target_silhouettes = render_silhouettes(target_mesh, cameras, renderer)
    print(f"      剪影形状: {tuple(target_silhouettes.shape)}, "
          f"均值覆盖率: {target_silhouettes.mean().item():.3f}")

    # 3. 初始化球体 + 可微顶点偏移量 ─────────────────────────────────────────
    print("[3/5] 初始化 ico_sphere(level={})...".format(CFG["sphere_level"]))
    src_mesh = ico_sphere(CFG["sphere_level"], device)
    n_verts = src_mesh.verts_packed().shape[0]
    print(f"      球体顶点数: {n_verts}, 三角面数: {src_mesh.faces_packed().shape[0]}")

    # deform_verts 是我们要优化的参数，初始化为全零（即不改变球体形状）
    deform_verts = torch.zeros((n_verts, 3), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([deform_verts], lr=CFG["lr"])
    # 每 100 步学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # 4. 优化主循环 ───────────────────────────────────────────────────────────
    print(f"[4/5] 开始优化（{CFG['num_iter']} 步）...\n")
    loss_log = {"total": [], "sil": [], "lap": [], "edge": [], "normal": []}
    frames = []

    for step in tqdm(range(CFG["num_iter"]), desc="优化进度"):
        optimizer.zero_grad()

        # 将偏移量加到球体顶点上，得到当前形变后的网格
        new_mesh = src_mesh.offset_verts(deform_verts)

        # 剪影损失：当前预测剪影 vs 目标剪影的 MSE
        pred_sil = render_silhouettes(new_mesh, cameras, renderer)
        loss_sil = torch.mean((pred_sil - target_silhouettes) ** 2)

        # 网格正则化损失
        loss_lap    = mesh_loss.mesh_laplacian_smoothing(new_mesh, method="uniform")
        loss_edge   = mesh_loss.mesh_edge_loss(new_mesh)
        loss_normal = mesh_loss.mesh_normal_consistency(new_mesh)

        # 总损失
        loss_total = (
            loss_sil
            + CFG["w_lap"]    * loss_lap
            + CFG["w_edge"]   * loss_edge
            + CFG["w_normal"] * loss_normal
        )

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # 记录
        loss_log["total"].append(loss_total.item())
        loss_log["sil"].append(loss_sil.item())
        loss_log["lap"].append(loss_lap.item())
        loss_log["edge"].append(loss_edge.item())
        loss_log["normal"].append(loss_normal.item())

        # 定期保存可视化帧
        save_this_step = (step % CFG["save_interval"] == 0) or (step == CFG["num_iter"] - 1)
        if save_this_step:
            with torch.no_grad():
                vis_sil = render_silhouettes(new_mesh, cameras, renderer)
            path = save_comparison_frame(
                step, CFG["num_iter"],
                target_silhouettes, vis_sil,
                loss_total.item(), loss_sil.item(),
                CFG["save_dir"],
                vis_views=CFG["vis_views"],
            )
            frames.append(path)

    # 5. 保存结果 ─────────────────────────────────────────────────────────────
    print("\n[5/5] 保存结果...")
    save_loss_curve(loss_log, CFG["save_dir"])

    if frames:
        gif_path = os.path.join(CFG["save_dir"], "optimization.gif")
        with imageio.get_writer(gif_path, mode="I", fps=CFG["gif_fps"]) as writer:
            for f in frames:
                writer.append_data(imageio.imread(f))
        print(f"  GIF → {gif_path}")

    print(f"\n最终总损失: {loss_log['total'][-1]:.4f}")
    print(f"输出目录:   {os.path.abspath(CFG['save_dir'])}")


if __name__ == "__main__":
    main()
