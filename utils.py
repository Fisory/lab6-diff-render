"""
工具函数：相机设置、渲染器构建、网格归一化、数据下载
"""

import os
import urllib.request
import numpy as np
import torch

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    PointLights,
    BlendParams,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[设备] CPU（若迭代较慢，可适当减少 num_views 和 num_iter）")
    return device


def get_cow_mesh_path(save_dir: str = "data/cow_mesh") -> str:
    """
    优先使用 pytorch3d 内置奶牛网格；若不存在则从 Facebook CDN 下载。
    """
    os.makedirs(save_dir, exist_ok=True)
    local_obj = os.path.join(save_dir, "cow.obj")

    # 尝试内置路径
    try:
        import pytorch3d
        builtin = os.path.join(
            os.path.dirname(pytorch3d.__file__), "data", "cow_mesh", "cow.obj"
        )
        if os.path.exists(builtin):
            return builtin
    except Exception:
        pass

    # 下载到 data/cow_mesh/
    base_url = "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/"
    for fname in ["cow.obj", "cow.mtl", "cow_texture.png"]:
        fpath = os.path.join(save_dir, fname)
        if not os.path.exists(fpath):
            print(f"  下载 {fname} ...")
            urllib.request.urlretrieve(base_url + fname, fpath)
    return local_obj


def normalize_mesh(mesh):
    """将网格中心化并缩放，使其恰好嵌入单位球。"""
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)
    scale = max((verts - center).abs().max().item(), 1e-8)
    mesh = mesh.offset_verts(-center)
    mesh = mesh.scale_verts(1.0 / scale)
    return mesh


def setup_cameras(
    num_views: int = 20,
    dist: float = 2.7,
    device: str = "cpu",
) -> FoVPerspectiveCameras:
    """在球面上均匀分布 num_views 个相机视角。"""
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T)


def build_silhouette_renderer(
    cameras: FoVPerspectiveCameras,
    image_size: int = 128,
    sigma: float = 1e-4,
    faces_per_pixel: int = 50,
) -> MeshRenderer:
    """
    构建软剪影渲染器。
    blur_radius 由 sigma 推导，使边界处的遮挡概率约为 1e-4。
    """
    blend_params = BlendParams(sigma=sigma)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def build_phong_renderer(
    cameras: FoVPerspectiveCameras,
    image_size: int = 128,
    sigma: float = 1e-4,
    faces_per_pixel: int = 50,
    device: str = "cpu",
) -> MeshRenderer:
    """
    构建 SoftPhong 渲染器，输出 RGB 图像（用于纹理优化）。
    """
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    blend_params = BlendParams(sigma=sigma)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        ),
    )
