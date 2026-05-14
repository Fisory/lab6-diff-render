"""
实验六 Part 2（选做）：联合纹理优化
同时优化网格顶点坐标和顶点颜色，拟合奶牛的形状与 RGB 外观。

与 Part 1 的区别：
  - 额外引入 verts_rgb（顶点颜色）作为可微参数
  - 用 SoftPhongShader 渲染 RGB 图像，在剪影损失基础上加入 RGB MSE 损失
  - 两组参数（顶点位移 vs 颜色）使用不同学习率，颜色收敛更快

参考：
  https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import imageio
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.utils import ico_sphere
import pytorch3d.loss as mesh_loss

from utils import (
    get_device,
    get_cow_mesh_path,
    normalize_mesh,
    setup_cameras,
    build_silhouette_renderer,
    build_phong_renderer,
)

# ─────────────────────────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "num_views":       20,
    "image_size":      128,
    "sphere_level":    4,
    "num_iter":        500,
    "lr_verts":        5e-3,    # 顶点位移学习率（比 Part 1 小，因为同时优化颜色会让梯度噪声变大）
    "lr_color":        1e-1,    # 颜色学习率（颜色空间更平滑，可以用更大的步长）
    "sigma":           1e-4,
    "faces_per_pixel": 50,
    "w_lap":           0.1,
    "w_edge":          1.0,
    "w_normal":        0.01,
    "w_rgb":           1.0,     # RGB 重建损失权重
    "save_dir":        "output/texture",
    "save_interval":   100,
    "gif_fps":         3,
    "vis_views":       [0, 5, 10, 15],
}


# ─────────────────────────────────────────────────────────────────────────────
# 网格构建
# ─────────────────────────────────────────────────────────────────────────────

def build_textured_mesh(base_mesh, deform_verts, verts_rgb_raw):
    """
    用球体基础网格 + 顶点偏移 + 顶点颜色构建当前形变网格。
    verts_rgb_raw 经过 sigmoid 映射到 (0, 1)，保证颜色合法且全程可微。
    """
    new_verts = base_mesh.verts_packed() + deform_verts
    verts_rgb = torch.sigmoid(verts_rgb_raw)        # (1, V, 3)，映射到 (0,1)
    textures = TexturesVertex(verts_features=verts_rgb)
    return Meshes(
        verts=[new_verts],
        faces=base_mesh.faces_list(),
        textures=textures,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 渲染工具
# ─────────────────────────────────────────────────────────────────────────────

def render_silhouettes(mesh, cameras, renderer):
    images = renderer(mesh.extend(len(cameras)), cameras=cameras)
    return images[..., 3]


def render_rgb(mesh, cameras, renderer):
    images = renderer(mesh.extend(len(cameras)), cameras=cameras)
    return images[..., :3]


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_frames(step, num_iter, target_sil, vis_sil, target_rgb, vis_rgb,
                loss_info, save_dir, vis_views=None):
    """同时保存多视角剪影对比帧和 RGB 对比帧，返回两个路径。"""
    if vis_views is None:
        vis_views = [0]
    n = len(vis_views)
    title = (
        f"Step {step}/{num_iter}  |  Loss: {loss_info['total']:.4f}  |  "
        f"Sil: {loss_info['sil']:.4f}  |  RGB: {loss_info['rgb']:.4f}"
    )

    # 剪影对比（多视角）
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, None]
    fig.suptitle(title, fontsize=11)
    for col, vi in enumerate(vis_views):
        axes[0, col].imshow(target_sil[vi].cpu().numpy(), cmap="gray")
        axes[0, col].set_title(f"Target Sil (view {vi})", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(vis_sil[vi].cpu().numpy(), cmap="gray")
        axes[1, col].set_title(f"Predicted (view {vi})", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout()
    p_sil = os.path.join(save_dir, f"sil_{step:04d}.png")
    plt.savefig(p_sil, dpi=100, bbox_inches="tight")
    plt.close()

    # RGB 对比（多视角）
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, None]
    fig.suptitle(title, fontsize=11)
    for col, vi in enumerate(vis_views):
        axes[0, col].imshow(target_rgb[vi].cpu().clamp(0, 1).numpy())
        axes[0, col].set_title(f"Target RGB (view {vi})", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(vis_rgb[vi].cpu().clamp(0, 1).numpy())
        axes[1, col].set_title(f"Optimized (view {vi})", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout()
    p_rgb = os.path.join(save_dir, f"rgb_{step:04d}.png")
    plt.savefig(p_rgb, dpi=100, bbox_inches="tight")
    plt.close()

    return p_sil, p_rgb


def save_loss_curve(loss_log, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for k in ["total", "sil", "rgb"]:
        ax1.plot(loss_log[k], label=k)
    ax1.set_title("主损失")
    ax1.set_xlabel("迭代步数")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for k in ["lap", "edge", "normal"]:
        ax2.plot(loss_log[k], label=k)
    ax2.set_title("正则化损失")
    ax2.set_xlabel("迭代步数")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = get_device()

    # 1. 加载目标网格 ────────────────────────────────────────────────────────
    print("\n[1/5] 加载目标奶牛网格...")
    obj_path = get_cow_mesh_path()
    target_mesh = load_objs_as_meshes([obj_path], device=device)
    target_mesh = normalize_mesh(target_mesh)

    # 2. 设置相机和渲染器，渲染目标图像 ─────────────────────────────────────
    print("[2/5] 设置渲染器并渲染目标图像...")
    cameras = setup_cameras(CFG["num_views"], device=device)
    sil_renderer   = build_silhouette_renderer(
        cameras, CFG["image_size"], CFG["sigma"], CFG["faces_per_pixel"]
    )
    phong_renderer = build_phong_renderer(
        cameras, CFG["image_size"], CFG["sigma"], CFG["faces_per_pixel"], device=device
    )

    with torch.no_grad():
        target_sil = render_silhouettes(target_mesh, cameras, sil_renderer)
        target_rgb = render_rgb(target_mesh, cameras, phong_renderer)
    print(f"      剪影覆盖率: {target_sil.mean().item():.3f}, "
          f"RGB 均值亮度: {target_rgb.mean().item():.3f}")

    # 3. 初始化球体 + 可微参数 ────────────────────────────────────────────────
    print("[3/5] 初始化 ico_sphere(level={})...".format(CFG["sphere_level"]))
    src_mesh = ico_sphere(CFG["sphere_level"], device)
    n_verts = src_mesh.verts_packed().shape[0]

    # 顶点偏移量：初始为 0（不改变球体）
    deform_verts = torch.zeros((n_verts, 3), device=device, requires_grad=True)

    # 顶点颜色（logit 空间）：初始 0 → sigmoid(0) = 0.5（中性灰），避免 clamp 梯度消失
    verts_rgb_raw = torch.zeros((1, n_verts, 3), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [deform_verts],  "lr": CFG["lr_verts"]},
        {"params": [verts_rgb_raw], "lr": CFG["lr_color"]},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    # 4. 优化主循环 ───────────────────────────────────────────────────────────
    print(f"[4/5] 开始联合优化（{CFG['num_iter']} 步）...\n")
    loss_log = {"total": [], "sil": [], "rgb": [], "lap": [], "edge": [], "normal": []}
    frames_sil, frames_rgb = [], []

    for step in tqdm(range(CFG["num_iter"]), desc="联合优化"):
        optimizer.zero_grad()

        new_mesh = build_textured_mesh(src_mesh, deform_verts, verts_rgb_raw)

        # 剪影损失
        pred_sil = render_silhouettes(new_mesh, cameras, sil_renderer)
        loss_sil = torch.mean((pred_sil - target_sil) ** 2)

        # RGB 重建损失
        pred_rgb = render_rgb(new_mesh, cameras, phong_renderer)
        loss_rgb = torch.mean((pred_rgb - target_rgb) ** 2)

        # 网格正则化
        loss_lap    = mesh_loss.mesh_laplacian_smoothing(new_mesh, method="uniform")
        loss_edge   = mesh_loss.mesh_edge_loss(new_mesh)
        loss_normal = mesh_loss.mesh_normal_consistency(new_mesh)

        loss_total = (
            loss_sil
            + CFG["w_rgb"]    * loss_rgb
            + CFG["w_lap"]    * loss_lap
            + CFG["w_edge"]   * loss_edge
            + CFG["w_normal"] * loss_normal
        )

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        loss_log["total"].append(loss_total.item())
        loss_log["sil"].append(loss_sil.item())
        loss_log["rgb"].append(loss_rgb.item())
        loss_log["lap"].append(loss_lap.item())
        loss_log["edge"].append(loss_edge.item())
        loss_log["normal"].append(loss_normal.item())

        save_this_step = (step % CFG["save_interval"] == 0) or (step == CFG["num_iter"] - 1)
        if save_this_step:
            with torch.no_grad():
                vis_sil = render_silhouettes(new_mesh, cameras, sil_renderer)
                vis_rgb = render_rgb(new_mesh, cameras, phong_renderer)

            p_sil, p_rgb = save_frames(
                step, CFG["num_iter"],
                target_sil, vis_sil,
                target_rgb, vis_rgb,
                {"total": loss_total.item(), "sil": loss_sil.item(), "rgb": loss_rgb.item()},
                CFG["save_dir"],
                vis_views=CFG["vis_views"],
            )
            frames_sil.append(p_sil)
            frames_rgb.append(p_rgb)

    # 5. 保存结果 ─────────────────────────────────────────────────────────────
    print("\n[5/5] 保存结果...")
    save_loss_curve(loss_log, CFG["save_dir"])

    for gif_name, frames in [
        ("sil_optimization.gif", frames_sil),
        ("rgb_optimization.gif", frames_rgb),
    ]:
        if frames:
            gif_path = os.path.join(CFG["save_dir"], gif_name)
            with imageio.get_writer(gif_path, mode="I", fps=CFG["gif_fps"]) as writer:
                for f in frames:
                    writer.append_data(imageio.imread(f))
            print(f"  GIF → {gif_path}")

    print(f"\n联合优化完成，最终损失: {loss_log['total'][-1]:.4f}")
    print(f"输出目录: {os.path.abspath(CFG['save_dir'])}")


if __name__ == "__main__":
    main()
