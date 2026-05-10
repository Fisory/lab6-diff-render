# 实验六：可微光栅化——把球体"捏"成奶牛

## 效果展示

【视频1:剪影优化过程——球体 300 步迭代逐渐变形为奶牛轮廓（多视角）】

【视频2:联合纹理优化——形状与顶点颜色同步收敛，从灰球变成有颜色的奶牛】

---

## 实验目标

- 搞清楚软光栅化（Soft Rasterization）是怎么解决梯度消失的，以及 σ 参数的直观含义
- 用 20 个视角的二维剪影图反推三维网格的顶点位置
- 理解网格正则化的作用——不加正则化网格会变成"刺猬"，三种正则项各自对应什么问题
- 选做：在形状优化的基础上同时拟合 RGB 外观（联合纹理优化）

---

## 原理

### 软光栅化

传统硬光栅化给每个像素返回 0 或 1（在三角形外还是里），在边界处的导数是 0，优化器没有任何信号。

软光栅化的做法：把"在不在三角形里面"这个硬判断，换成基于**带符号距离** $d$ 的连续概率：

$$A(d) = \sigma\!\left(\frac{d}{\sigma}\right) = \frac{1}{1 + e^{-d/\sigma}}$$

$d > 0$ 表示像素在三角形内，$d < 0$ 在外面。$\sigma$ 控制边缘的模糊程度——$\sigma$ 越小越接近硬光栅化，$\sigma$ 越大边缘越模糊。关键在于，sigmoid 在 $d=0$（边界）处有非零导数，顶点无论在哪里都能收到梯度。

代码中的 `blur_radius` 参数由 `sigma` 推导，确保概率低于 `1e-4` 的像素被排除（省去无效计算）：

```python
blur_radius = np.log(1. / 1e-4 - 1.) * sigma
```

### 网格正则化

只用剪影损失优化顶点会有个严重问题：顶点为了迎合某个视角的投影会疯狂移动，最终网格严重扭曲甚至三角面穿插。加三项正则化来约束：

| 正则化项 | 用途 | 权重 |
|---|---|---|
| **拉普拉斯平滑** $L_{lap}$ | 每个顶点向邻居平均位置靠拢，抑制局部尖刺 | `w_lap = 0.1` |
| **边长一致性** $L_{edge}$ | 惩罚过长/过短的边，防止三角形退化 | `w_edge = 1.0` |
| **法线一致性** $L_{normal}$ | 相邻面法线方向要相近，保持表面平滑 | `w_normal = 0.01` |

总损失：

$$L_{\text{total}} = L_{\text{sil}} + w_{\text{lap}} L_{\text{lap}} + w_{\text{edge}} L_{\text{edge}} + w_{\text{normal}} L_{\text{normal}}$$

`w_edge = 1.0` 明显大于其他项，因为边长崩坏是最容易出现的拓扑问题，尤其在迭代早期。

---

## 项目结构

```
lab6_diff_render/
├── utils.py              # 渲染器构建、相机设置、网格归一化、数据下载
├── main_silhouette.py    # Part 1：剪影优化（球 → 奶牛形状）
├── main_texture.py       # Part 2（选做）：联合纹理优化（形状 + 颜色）
├── pyproject.toml        # uv 依赖配置（torch 2.4.0 CPU）
└── output/               # 运行后自动生成
    ├── silhouette/       # Part 1 输出帧、loss 曲线、GIF
    └── texture/          # Part 2 输出帧、loss 曲线、GIF
```

---

## 环境配置

pytorch3d 目前（0.7.8）最高支持 PyTorch 2.4.x，所以**不要用系统里的 torch 2.6**，需要单独建环境。

### 方法一：Conda（Windows 推荐）

```bash
conda create -n p3d python=3.11
conda activate p3d
conda install pytorch==2.4.0 torchvision==0.19.0 cpuonly -c pytorch
conda install -c pytorch3d pytorch3d
pip install matplotlib tqdm imageio pillow scipy
```

### 方法二：uv + pip

```bash
# 安装基础依赖（torch 2.4.0 CPU 版 + 其他包）
cd lab6_diff_render
uv sync

# pytorch3d 预编译 wheel（仅 Linux，Python 3.11 + CPU + torch 2.4.0）
pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cpu_pyt240/download.html

# Windows 需从源码编译（需要 MSVC Build Tools 17+）
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

> **验证安装**
> ```python
> import pytorch3d; print(pytorch3d.__version__)  # 期望输出 0.7.x
> ```

---

## 运行方法

```bash
# Part 1：剪影优化（约 300 步，CPU 上大概跑 10~30 分钟）
python main_silhouette.py

# Part 2（选做）：联合纹理优化（500 步，比 Part 1 慢一倍左右）
python main_texture.py
```

第一次运行会自动从 Facebook CDN 下载奶牛网格文件（`cow.obj` + 纹理，约 2 MB）到 `data/cow_mesh/`。

输出文件说明：
- `output/silhouette/frame_XXXX.png` — 每 50 步的剪影对比截图
- `output/silhouette/loss_curve.png` — 各项损失随迭代的变化曲线
- `output/silhouette/optimization.gif` — 完整优化过程动画

---

## 实验结果

### Part 1：剪影优化

【视频3:损失曲线截图——total / sil / lap / edge / normal 各项随迭代的变化】

迭代 300 步后，从 20 个视角看，球体的轮廓已经和奶牛剪影比较接近，能看出躯干、腿部、头部的大致轮廓。最终总损失降到 0.02 左右，剪影误差约 0.014（与参考效果基本一致）。

**关键参数影响：**

- **`sigma`**：设为 `1e-4` 是个比较平衡的选择。太大（如 `1e-2`）边缘过于模糊，剪影分辨率会降低；太小则接近硬光栅化，梯度消失，迭代没有效果。
- **`w_lap`**：拉普拉斯权重太大网格过于平滑，细节消失；太小会出现局部尖刺。`0.1` 是个合理默认值。
- **`sphere_level`**：用 4 级细分（2562 个顶点），顶点数足够表达奶牛的细节，但 CPU 上也还能跑。如果显存/内存充足可以试试 level 5（10242 顶点）。
- **学习率调度**：每 100 步衰减到原来的 0.5，前期收敛快，后期精细调整。

### Part 2（选做）：联合纹理优化

【视频4:RGB 优化过程——顶点颜色从灰色逐渐拟合奶牛的棕白色外观】

在 Part 1 的基础上新增了：
- `verts_rgb_raw`（顶点颜色，logit 空间）作为额外的可微参数
- `SoftPhongShader` 渲染 RGB 图像，计算与目标 RGB 的 MSE
- 颜色用 `sigmoid` 映射到 `(0,1)`，避免 `clamp` 在边界处梯度为零的问题

颜色收敛比形状快，通常 100~150 步内颜色就趋近目标了，但形状还需要更多步数。两组参数使用不同学习率（顶点位移 `5e-3`，颜色 `1e-1`）放在同一个 Adam 里。

---

## 代码说明

### `utils.py`

三个核心函数：

**`setup_cameras`**：在球面上均匀分布 N 个视角，`linspace` 等间距采样仰角和方位角，用 `look_at_view_transform` 生成旋转矩阵。多视角是这个方法的关键——单视角会导致对称性歧义（从正前方看，左右两侧是对称的，优化器分不清）。

**`build_silhouette_renderer`**：把 `blur_radius`、`faces_per_pixel`、`BlendParams(sigma)` 组合起来，构建 `SoftSilhouetteShader`。`faces_per_pixel=50` 表示每个像素最多考虑 50 个重叠的三角面，数值越大渲染越准确但越慢。

**`normalize_mesh`**：先减均值中心化，再除以最大坐标幅度，把网格缩放到单位球内。不归一化的话不同模型的尺度差异很大，相机距离参数难以统一设置。

### `main_silhouette.py`

核心优化循环：

```python
deform_verts = torch.zeros((n_verts, 3), device=device, requires_grad=True)

for step in range(num_iter):
    new_mesh = src_mesh.offset_verts(deform_verts)   # 形变
    pred_sil  = render_silhouettes(new_mesh, cameras, renderer)
    loss_sil  = torch.mean((pred_sil - target_silhouettes) ** 2)
    loss_total = loss_sil + w_lap*loss_lap + w_edge*loss_edge + w_normal*loss_normal
    loss_total.backward()
    optimizer.step()
```

`src_mesh.offset_verts(deform_verts)` 是 pytorch3d 提供的 in-place 形变接口，不改变原始球体，直接返回新的 `Meshes` 对象。`mesh.extend(n)` 把单个网格复制 n 份，配合 n 个相机一起批量渲染，省去了手动循环。

### `main_texture.py`

相比 Part 1 增加了顶点颜色的优化：

```python
verts_rgb_raw = torch.zeros((1, n_verts, 3), device=device, requires_grad=True)
# 在每步循环中：
verts_rgb = torch.sigmoid(verts_rgb_raw)   # 映射到 (0,1)
textures  = TexturesVertex(verts_features=verts_rgb)
new_mesh  = Meshes(verts=[...], faces=[...], textures=textures)
```

用 logit 空间参数化颜色（而不是直接对颜色值做 `clamp`），可以保证全程有非零梯度。

---

## 参考资料

- [pytorch3d fit_textured_mesh notebook](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb)
- [pytorch3d deform_source_mesh_to_target_mesh notebook](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)
- Liu et al. "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning" ICCV 2019
- Loper & Black "OpenDR: An Approximate Differentiable Renderer" ECCV 2014

---

北京师范大学 · 计算机图形学实验
