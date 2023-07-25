# adapted from https://github.com/EPFL-VILAB/omnidata
import sys
from pathlib import Path
import argparse
import math
from PIL import Image, ImageOps

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

DEBUG = True

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_patch_path = Path("/home/dawars/personal_projects/sdfstudio/data/heritage/nepszinhaz_internal/patches")
output_patch_path.mkdir(exist_ok=True)

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


# TODO merge the following 4 function to one single function

# align depth map in the x direction from left to right
def align_x(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[1] == depth2.shape[1]

    assert (e1 - s1) == (e2 - s2)  # 288== 192
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, :, s2:e2], depth1[:, :, s1:e1],
                                           torch.ones_like(depth1[:, :, s1:e1]))

    depth2_aligned = scale * depth2 + shift
    result = torch.ones((1, depth1.shape[1], depth1.shape[2] + depth2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = depth1[:, :, :s1]
    result[:, :, depth1.shape[2]:] = depth2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1 - s1))[None, None, :]
    result[:, :, s1:depth1.shape[2]] = depth1[:, :, s1:] * weight + depth2_aligned[:, :, :e2] * (1 - weight)

    return result


# align depth map in the y direction from top to down
def align_y(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[2] == depth2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, s2:e2, :], depth1[:, s1:e1, :],
                                           torch.ones_like(depth1[:, s1:e1, :]))

    depth2_aligned = scale * depth2 + shift
    result = torch.ones((1, depth1.shape[1] + depth2.shape[1] - (e1 - s1), depth1.shape[2]))

    result[:, :s1, :] = depth1[:, :s1, :]
    result[:, depth1.shape[1]:, :] = depth2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1 - s1))[None, :, None]
    result[:, s1:depth1.shape[1], :] = depth1[:, s1:, :] * weight + depth2_aligned[:, :e2, :] * (1 - weight)

    return result


# align normal map in the x direction from left to right
def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)

    R = best_fit_transform(normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2]:] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1 - s1))[None, None, :]

    result[:, :, s1:normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[:, :, :e2] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]

    return result


# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)

    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1 - s1))[None, :, None]

    result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]

    return result


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


offset_fraction = 3  # determines overlap

size = 384
size2 = size // 2
offset = size // offset_fraction  # a lot of overlap, slower but pixel perfect division, bad global value distribution


def inference_patch(image: Image, device: torch.device = "cpu"):
    """
    image: 3 channel PIL Image
    """
    with torch.no_grad():
        img_tensor_depth = trans_depth(image)[:3].unsqueeze(0)
        img_tensor_normal = trans_normal(image)[:3].unsqueeze(0)

        # if img_tensor_normal.shape[1] == 1:
        #     img_tensor_normal = img_tensor_normal.repeat_interleave(3, 1)

        depth = net_depth(img_tensor_depth.cuda()).clamp(min=0, max=1).to(device)
        normal = net_normal(img_tensor_normal.cuda()).clamp(min=0, max=1).to(device)

        return depth, normal


def process_image(image_path: Path, out_dir: Path):
    image_name = image_path.name
    out_path_normal = out_dir / "normal" / image_name
    out_path_depth = out_dir / "depth" / image_name

    patches = {}
    _image = Image.open(image_path).convert("RGB")
    orig_size = _image.size

    _image.thumbnail((1400, 1400))
    W, H = _image.size

    x = math.ceil(W / offset)
    y = math.ceil(H / offset)

    is_too_small = H < size and W < size
    if is_too_small:
        if DEBUG:
            print(f"Image too small {image_path}")
        _image = ImageOps.pad(_image, (size, size), Resampling.BICUBIC, centering=(0, 0))
        x = y = 2
    else:
        # padding so that whole image can be recovered
        _image = ImageOps.pad(_image, (x * offset, y * offset), Resampling.BICUBIC, centering=(0, 0))
    image_cv = cv2.cvtColor(np.array(_image), cv2.COLOR_RGB2BGR)

    normal_patches = {}
    depth_patches = {}

    for j in range(y - 1):
        for i in range(x - 1):
            # crop images
            image_patch = image_cv[j * offset : j * offset + size, i * offset : i * offset + size, :]
            # if DEBUG:
            #     patches[(i, j)] = image_patch

                # image_patch.save()
            # inference
            depth, normal = inference_patch(image_patch)

            depth_patches[(i, j)] = depth
            normal_patches[(i, j)] = normal[0]

    if is_too_small:
        depth_top = depth_patches[(0, 0)]
        depth_top = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min())
        depth_top = depth_top[:, :H, :W]

        plt.imsave(out_path_depth.with_suffix('.png'), depth_top[0].numpy(), cmap='viridis')
        np.save(out_path_depth.with_suffix('.npy'), depth_top.detach().cpu().numpy()[0])

        normal_top = normal_patches[(0, 0)].numpy()
        normal_top = normal_top[:, :H, :W]

        plt.imsave(out_path_normal.with_suffix('.png'), np.moveaxis(normal_top, [0, 1, 2], [2, 0, 1]))
        np.save(out_path_normal.with_suffix('.npy'), normal_top)

        return

    # save middle file for alignments
    start_y = max(0, offset * (y // offset_fraction) - size2)
    start_x = max(0, offset * (x // offset_fraction) - size2)
    image_middle = image_cv[start_y : start_y + size, start_x : start_x + size]
    depth_middle, normal_middle = inference_patch(image_middle)

    depths_row = []
    # align depth maps from left to right row by row
    for j in range(y - 1):
        depths = []
        for i in range(x - 1):
            depth = depth_patches[(i, j)]
            depths.append(depth)

        # align from left to right
        depth_left = depths[0]
        s1 = offset
        s2 = 0
        e2 = size - offset
        for depth_right in depths[1:]:
            depth_left = align_x(depth_left, depth_right, s1, depth_left.shape[2], s2, e2)
            s1 += offset
        depths_row.append(depth_left)
        # print(depth_left.shape)

    depth_top = depths_row[0]
    # align depth maps from top to down
    s1 = offset
    s2 = 0
    e2 = size - offset
    for depth_bottom in depths_row[1:]:
        depth_top = align_y(depth_top, depth_bottom, s1, depth_top.shape[1], s2, e2)
        s1 += offset

    # depth is up to scale so don't need to align to middle part
    start_y = max(0, offset * (y // offset_fraction) - size2)
    start_x = max(0, offset * (x // offset_fraction) - size2)
    scale, shift = compute_scale_and_shift(
        depth_top[:, start_y:start_y + size, start_x:start_x + size],
        depth_middle,
        torch.ones_like(depth_middle))  # todo mask for sky region
    depth_top = scale * depth_top + shift

    depth_top = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min())
    depth_top = depth_top[:, :H, :W]
    depth_top = depth_top[0].numpy()

    if orig_size != (W, H):
        depth_top = np.array(Image.fromarray(depth_top).resize(orig_size, resample=Resampling.BICUBIC))

    if DEBUG:
        plt.imsave(out_path_depth.with_suffix(".viz.png"), depth_top, cmap="viridis")

    np.save(out_path_depth.with_suffix(".npy"), depth_top)
    depth_top_cv = np.uint16(depth_top * (2 ** 16 - 1))
    cv2.imwrite(str(out_path_depth.with_suffix('.png')), depth_top_cv)

    # normal
    normals_row = []
    # align normal maps from left to right row by row
    for j in range(y - 1):
        normals = []
        for i in range(x - 1):
            normal = normal_patches[(i, j)].numpy()
            normal = normal * 2. - 1.
            normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
            normals.append(normal)

        # align from left to right
        normal_left = normals[0]
        s1 = offset
        s2 = 0
        e2 = size - offset
        for normal_right in normals[1:]:
            normal_left = align_normal_x(normal_left, normal_right, s1, normal_left.shape[2], s2, e2)
            s1 += offset
        normals_row.append(normal_left)
        # print(normal_left.shape)

    normal_top = normals_row[0]
    # align normal maps from top to down
    s1 = offset
    s2 = 0
    e2 = size - offset
    for normal_bottom in normals_row[1:]:
        # print(normal_top.shape, normal_bottom.shape)
        normal_top = align_normal_y(normal_top, normal_bottom, s1, normal_top.shape[1], s2, e2)
        s1 += offset

    # align to middle part
    mid_normal = normal_middle
    mid_normal = mid_normal * 2. - 1.
    mid_normal = mid_normal / (np.linalg.norm(mid_normal, axis=0) + 1e-15)[None]

    start_y = max(0, offset * (y // offset_fraction) - size2)
    start_x = max(0, offset * (x // offset_fraction) - size2)
    R = best_fit_transform(
        normal_top[:, start_y : start_y + size, start_x : start_x + size].reshape(3, -1).T, mid_normal.reshape(3, -1).T, mask.T
    )
    normal_top = (R @ normal_top.reshape(3, -1)).reshape(normal_top.shape)

    normal_top = normal_top[:, :H, :W]
    normal_top = (normal_top + 1.0) / 2.0
    normal_top = np.transpose(normal_top, [1, 2, 0])  # HWC float opencv

    if orig_size != (W, H):  # todo test this
        normal_top = cv2.resize(normal_top, orig_size, interpolation=cv2.INTER_NEAREST)

    if DEBUG:
        cv2.imwrite(str(out_path_normal.with_suffix(".png")), np.uint8(normal_top[..., [2, 1, 0]] * 255))
        # normal_top.save(out_path_normal.with_suffix('.png'))

    normal_top = np.transpose(normal_top, [2, 0, 1])  # HWC -> CHW
    np.save(out_path_normal.with_suffix(".npy"), normal_top)  # [0,1] float


def load_model(mode: str):
    from modules.midas.dpt_depth import DPTDepthModel

    if mode in ["normal", "depth"]:
        pretrained_weights_path = root_dir + f"omnidata_dpt_{mode}_v2.ckpt"
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=1 if mode == "depth" else 3)  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        return model


def process_scene(data_root: Path, scene: str, out_path_prefix: Path):
    """ """
    print(f"Processing scene {scene}")

    image_dir = "dense/images"  # relative dir to data_root/{scene}/
    paths = []

    extensions = ["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]
    for ext in extensions:
        paths.extend((data_root / scene / image_dir).glob(f"[!.]*.{ext}"))
    paths = sorted(paths)
    out_path = out_path_prefix / scene

    (out_path / "normal").mkdir(parents=True, exist_ok=True)
    (out_path / "depth").mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(paths):
        process_image(image_path, out_path)


def main(args):
    data_root = Path(args.input_path)

    out_path_prefix = Path(args.output_path)  # result will be saved to path/scene/{normal|depth}
    scenes = ['brandenburg_gate', 'pantheon_exterior']

    for scene in scenes:
        process_scene(data_root, scene, out_path_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize output for depth or surface normals")

    parser.add_argument("--omnidata-path", default="../omnidata/omnidata_tools/torch/", help="path to omnidata model")

    parser.add_argument(
        "--pretrained-models", default="../omnidata/omnidata_tools/torch/pretrained_models/", help="path to pretrained models"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="/home/dawars/personal_projects/sdfstudio/data/heritage/",
        help="Path to root of phototourism datasets",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/dawars/personal_projects/sdfstudio/data/heritage/",
        help="path to where output images should be stored (output_path/{scene}/{depth/normal}",
    )
    args = parser.parse_args()

    # download to args.pretrained_models
    # gdown '1iJjV9rkdeLvsTU9x3Vx8vwZUg-sSQ9nm&confirm=t'  # omnidata normals (v1)
    # gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t'  # omnidata normals (v2)
    # gdown '1UxUDbEygQ-CMBjRKACw_Xdj4RkDjirB5&confirm=t'  # omnidata depth (v1)
    # gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t'  # omnidata depth (v2)

    root_dir = args.pretrained_models
    omnidata_path = args.omnidata_path

    sys.path.append(args.omnidata_path)
    from data.transforms import get_transform

    net_normal = load_model("normal")
    net_depth = load_model("depth")

    trans_normal = transforms.Compose([get_transform("rgb", image_size=None)])
    trans_depth = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    trans_topil = transforms.ToPILImage()

    main(args)
