# %% [markdown]
# # 实验六：可微光栅化——把球体「捏」成奶牛
#
# **在 Google Colab 上运行步骤：**
# 1. 上传本文件到 Colab（或 `!wget` 直接拉取）
# 2. Runtime → Change runtime type → T4 GPU（可选）
# 3. 按顺序执行每个 `# %%` 单元

# %% 安装依赖
# ─────────────────────────────────────────────────────────────────────────────
import sys, os

# 检测运行环境
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import torch
    pyt = torch.__version__.split('+')[0].replace('.', '')
    cuda = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
    py  = f'py3{sys.version_info.minor}'
    ver = f'{py}_{cuda}_pyt{pyt}'
    print(f'版本标识: {ver}')
    os.system('pip install fvcore iopath -q')
    os.system(
        f'pip install --no-index --no-cache-dir pytorch3d '
        f'-f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{ver}/download.html'
    )
    os.system('pip install imageio tqdm matplotlib -q')

# %% 导入
# ─────────────────────────────────────────────────────────────────────────────
import urllib.request
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import tqdm

from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftSilhouetteShader, SoftPhongShader,
    PointLights, BlendParams, TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes
import pytorch3d.loss as mesh_loss

print('pytorch3d 版本:', __import__('pytorch3d').__version__)


# %% 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'[设备] {dev}')
    return dev


def download_cow(save_dir='cow_mesh'):
    os.makedirs(save_dir, exist_ok=True)
    base = 'https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/'
    for f in ['cow.obj', 'cow.mtl', 'cow_texture.png']:
        p = os.path.join(save_dir, f)
        if not os.path.exists(p):
            print(f'  下载 {f}...')
            urllib.request.urlretrieve(base + f, p)
    return os.path.join(save_dir, 'cow.obj')


def normalize_mesh(mesh):
    v = mesh.verts_packed()
    c = v.mean(0)
    s = max((v - c).abs().max().item(), 1e-8)
    return mesh.offset_verts(-c).scale_verts(1.0 / s)


def setup_cameras(n=20, dist=2.7, device='cpu'):
    elev = torch.linspace(0, 360, n)
    azim = torch.linspace(-180, 180, n)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T)


def build_sil_renderer(cameras, size=128, sigma=1e-4, fpp=50):
    bp = BlendParams(sigma=sigma)
    rs = RasterizationSettings(
        image_size=size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=fpp,
    )
    return MeshRenderer(
        MeshRasterizer(cameras=cameras, raster_settings=rs),
        SoftSilhouetteShader(blend_params=bp),
    )


def build_phong_renderer(cameras, size=128, sigma=1e-4, fpp=50, device='cpu'):
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    bp = BlendParams(sigma=sigma)
    rs = RasterizationSettings(
        image_size=size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=fpp,
    )
    return MeshRenderer(
        MeshRasterizer(cameras=cameras, raster_settings=rs),
        SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=bp),
    )


print('工具函数就绪')


# %% [markdown]
# ## Part 1：剪影优化（球体 → 奶牛形状）

# %% Part 1 超参数
# ─────────────────────────────────────────────────────────────────────────────
NUM_VIEWS   = 20
IMAGE_SIZE  = 128
NUM_ITER    = 500
LR          = 1e-2
SIGMA       = 1e-4
W_LAP       = 0.1
W_EDGE      = 1.0
W_NORMAL    = 0.01
VIS_VIEWS   = [0, 5, 10, 15]   # 可视化时展示的 4 个代表性视角
SAVE_DIR    = 'output/silhouette'
os.makedirs(SAVE_DIR, exist_ok=True)

device = get_device()


# %% 加载目标网格 + 渲染目标剪影
# ─────────────────────────────────────────────────────────────────────────────
obj_path    = download_cow()
target_mesh = normalize_mesh(load_objs_as_meshes([obj_path], device=device))
cameras     = setup_cameras(NUM_VIEWS, device=device)
renderer    = build_sil_renderer(cameras, IMAGE_SIZE, SIGMA)

with torch.no_grad():
    target_sil = renderer(target_mesh.extend(NUM_VIEWS), cameras=cameras)[..., 3]

# 展示目标剪影（4 个视角）
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(target_sil[i * 5].cpu().numpy(), cmap='gray')
    ax.set_title(f'视角 {i * 5}')
    ax.axis('off')
plt.suptitle('目标奶牛剪影（4 个视角）')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/target_silhouettes.png', dpi=100)
plt.show()
print(f'目标剪影形状: {tuple(target_sil.shape)}')


# %% 优化主循环
# ─────────────────────────────────────────────────────────────────────────────
src_mesh = ico_sphere(4, device)
n_verts  = src_mesh.verts_packed().shape[0]
print(f'球体: {n_verts} 顶点, {src_mesh.faces_packed().shape[0]} 面')

deform_verts = torch.zeros((n_verts, 3), device=device, requires_grad=True)
optimizer    = torch.optim.Adam([deform_verts], lr=LR)
scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

loss_log = {'total': [], 'sil': [], 'lap': [], 'edge': [], 'normal': []}
frames   = []

for step in tqdm(range(NUM_ITER), desc='剪影优化'):
    optimizer.zero_grad()
    new_mesh = src_mesh.offset_verts(deform_verts)

    pred_sil    = renderer(new_mesh.extend(NUM_VIEWS), cameras=cameras)[..., 3]
    loss_sil    = torch.mean((pred_sil - target_sil) ** 2)
    loss_lap    = mesh_loss.mesh_laplacian_smoothing(new_mesh, method='uniform')
    loss_edge   = mesh_loss.mesh_edge_loss(new_mesh)
    loss_normal = mesh_loss.mesh_normal_consistency(new_mesh)

    loss_total = (
        loss_sil
        + W_LAP    * loss_lap
        + W_EDGE   * loss_edge
        + W_NORMAL * loss_normal
    )
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    loss_log['total'].append(loss_total.item())
    loss_log['sil'].append(loss_sil.item())
    loss_log['lap'].append(loss_lap.item())
    loss_log['edge'].append(loss_edge.item())
    loss_log['normal'].append(loss_normal.item())

    if step % 50 == 0 or step == NUM_ITER - 1:
        with torch.no_grad():
            vis = renderer(new_mesh.extend(NUM_VIEWS), cameras=cameras)[..., 3]

        n_v = len(VIS_VIEWS)
        fig, axes = plt.subplots(2, n_v, figsize=(4 * n_v, 8))
        fig.suptitle(
            f'Step {step}/{NUM_ITER}  |  Loss: {loss_total.item():.4f}  |  Sil: {loss_sil.item():.4f}',
            fontsize=12,
        )
        for col, vi in enumerate(VIS_VIEWS):
            axes[0, col].imshow(target_sil[vi].cpu().numpy(), cmap='gray')
            axes[0, col].set_title(f'Target (view {vi})', fontsize=9)
            axes[0, col].axis('off')
            axes[1, col].imshow(vis[vi].cpu().numpy(), cmap='gray')
            axes[1, col].set_title(f'Predicted (view {vi})', fontsize=9)
            axes[1, col].axis('off')
        plt.tight_layout()
        p = f'{SAVE_DIR}/frame_{step:04d}.png'
        plt.savefig(p, dpi=100, bbox_inches='tight')
        frames.append(p)
        plt.close()

final_loss = loss_log['total'][-1]
print(f'优化完成，最终损失: {final_loss:.4f}')


# %% 保存 GIF + 损失曲线
# ─────────────────────────────────────────────────────────────────────────────
gif_sil = f'{SAVE_DIR}/optimization.gif'
with imageio.get_writer(gif_sil, mode='I', fps=5) as w:
    for f in frames:
        w.append_data(imageio.imread(f))
print(f'GIF 已保存: {gif_sil}')

# 展示最后一帧
plt.figure(figsize=(10, 5))
plt.imshow(imageio.imread(frames[-1]))
plt.axis('off')
plt.tight_layout()
plt.show()

# 损失曲线
fig, ax = plt.subplots(figsize=(10, 4))
for k, v in loss_log.items():
    ax.plot(v, label=k)
ax.set_xlabel('迭代步数')
ax.set_ylabel('Loss')
ax.set_title('剪影优化损失曲线')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/loss_curve.png', dpi=120)
plt.show()


# %% [markdown]
# ## Part 2（选做）：联合纹理优化

# %% Part 2 超参数
# ─────────────────────────────────────────────────────────────────────────────
NUM_ITER2 = 500
LR_VERTS  = 5e-3
LR_COLOR  = 1e-1
W_RGB     = 1.0
SAVE_DIR2 = 'output/texture'
os.makedirs(SAVE_DIR2, exist_ok=True)


# %% 渲染目标 RGB
# ─────────────────────────────────────────────────────────────────────────────
p_renderer = build_phong_renderer(cameras, IMAGE_SIZE, SIGMA, device=device)

with torch.no_grad():
    target_rgb = p_renderer(target_mesh.extend(NUM_VIEWS), cameras=cameras)[..., :3]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(target_rgb[i * 5].cpu().clamp(0, 1).numpy())
    ax.set_title(f'视角 {i * 5}')
    ax.axis('off')
plt.suptitle('目标奶牛 RGB（4 个视角）')
plt.tight_layout()
plt.show()


# %% 联合优化主循环
# ─────────────────────────────────────────────────────────────────────────────
src2   = ico_sphere(4, device)
n2     = src2.verts_packed().shape[0]
dv     = torch.zeros((n2, 3), device=device, requires_grad=True)
cr_raw = torch.zeros((1, n2, 3), device=device, requires_grad=True)  # logit 空间颜色

r_sil2 = build_sil_renderer(cameras, IMAGE_SIZE, SIGMA)

opt2 = torch.optim.Adam([
    {'params': [dv],     'lr': LR_VERTS},
    {'params': [cr_raw], 'lr': LR_COLOR},
])
sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=150, gamma=0.5)

log2   = {'total': [], 'sil': [], 'rgb': [], 'lap': [], 'edge': [], 'normal': []}
fr_sil, fr_rgb = [], []

for step in tqdm(range(NUM_ITER2), desc='联合优化'):
    opt2.zero_grad()

    new_verts = src2.verts_packed() + dv
    tex  = TexturesVertex(verts_features=torch.sigmoid(cr_raw))
    nm   = Meshes(verts=[new_verts], faces=src2.faces_list(), textures=tex)

    ps   = r_sil2(nm.extend(NUM_VIEWS), cameras=cameras)[..., 3]
    pr   = p_renderer(nm.extend(NUM_VIEWS), cameras=cameras)[..., :3]

    l_sil = torch.mean((ps - target_sil) ** 2)
    l_rgb = torch.mean((pr - target_rgb) ** 2)
    l_lap = mesh_loss.mesh_laplacian_smoothing(nm, method='uniform')
    l_edg = mesh_loss.mesh_edge_loss(nm)
    l_nrm = mesh_loss.mesh_normal_consistency(nm)

    lt = l_sil + W_RGB * l_rgb + W_LAP * l_lap + W_EDGE * l_edg + W_NORMAL * l_nrm
    lt.backward()
    opt2.step()
    sch2.step()

    for k, v in zip(['total', 'sil', 'rgb', 'lap', 'edge', 'normal'],
                    [lt, l_sil, l_rgb, l_lap, l_edg, l_nrm]):
        log2[k].append(v.item())

    if step % 100 == 0 or step == NUM_ITER2 - 1:
        with torch.no_grad():
            vs = r_sil2(nm.extend(NUM_VIEWS), cameras=cameras)[..., 3]
            vr = p_renderer(nm.extend(NUM_VIEWS), cameras=cameras)[..., :3]

        title = (
            f'Step {step}/{NUM_ITER2} | '
            f'Loss:{lt.item():.4f}  Sil:{l_sil.item():.4f}  RGB:{l_rgb.item():.4f}'
        )

        # 剪影对比帧
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title)
        axes[0].imshow(target_sil[0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Target Silhouette')
        axes[0].axis('off')
        axes[1].imshow(vs[0].cpu().numpy(), cmap='gray')
        axes[1].set_title(f'Predicted (Step {step})')
        axes[1].axis('off')
        plt.tight_layout()
        p = f'{SAVE_DIR2}/sil_{step:04d}.png'
        plt.savefig(p, dpi=100, bbox_inches='tight')
        fr_sil.append(p)
        plt.close()

        # RGB 对比帧
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title)
        axes[0].imshow(target_rgb[0].cpu().clamp(0, 1).numpy())
        axes[0].set_title('Target RGB')
        axes[0].axis('off')
        axes[1].imshow(vr[0].cpu().clamp(0, 1).numpy())
        axes[1].set_title(f'Optimized RGB (Step {step})')
        axes[1].axis('off')
        plt.tight_layout()
        p = f'{SAVE_DIR2}/rgb_{step:04d}.png'
        plt.savefig(p, dpi=100, bbox_inches='tight')
        fr_rgb.append(p)
        plt.close()

print(f'联合优化完成，最终损失: {log2["total"][-1]:.4f}')


# %% 保存 GIF + 展示
# ─────────────────────────────────────────────────────────────────────────────
for name, flist in [('sil_optimization.gif', fr_sil), ('rgb_optimization.gif', fr_rgb)]:
    out = f'{SAVE_DIR2}/{name}'
    with imageio.get_writer(out, mode='I', fps=3) as w:
        for f in flist:
            w.append_data(imageio.imread(f))
    print(f'已保存: {out}')

# 展示最后一帧对比
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(imageio.imread(fr_sil[-1]))
axes[0].set_title('剪影优化最终帧')
axes[0].axis('off')
axes[1].imshow(imageio.imread(fr_rgb[-1]))
axes[1].set_title('RGB 优化最终帧')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# %% 在 Colab 上下载 GIF 到本地
# ─────────────────────────────────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import files
    files.download('output/silhouette/optimization.gif')
    files.download('output/texture/sil_optimization.gif')
    files.download('output/texture/rgb_optimization.gif')
