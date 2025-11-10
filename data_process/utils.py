import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def z_score_normalize(data):
    q1 = np.percentile(data, 1)
    q99 = np.percentile(data, 99)
    
    arr_clipped = np.where(data < q1, q1, data)
    arr_clipped = np.where(arr_clipped > q99, q99, arr_clipped)
    
    mean = np.mean(arr_clipped)
    std = np.std(arr_clipped)
    return (arr_clipped - mean) / std

def extract_patches(image, patch_size=128, stride=32):
    """
    从一张 [C, H, W] 图像中滑动裁剪 patch，确保 patch 不越界（不使用 padding）
    """
    print(f'extract_patches: {image.shape}')
    C, H, W = image.shape
    patches = []
    coords = []

    top_positions = list(range(0, H - patch_size + 1, stride))
    if top_positions[-1] != H - patch_size:
        top_positions.append(H - patch_size)

    left_positions = list(range(0, W - patch_size + 1, stride))
    if left_positions[-1] != W - patch_size:
        left_positions.append(W - patch_size)

    for top in top_positions:
        for left in left_positions:
            patch = image[:, top:top+patch_size, left:left+patch_size]
            patches.append(patch)
            coords.append((top, left))

    return patches, coords, H, W

def predict_full_image_batch(model, image, patch_size=150, stride=8, min_size_threshold=100, num_classes=4, cmap=None, is_shown=False):
    """
    分割模型滑窗预测（投票式argmax），输出最终 [H, W] 分割图
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = np.stack([image] * 3, axis=0)  # [C, H, W]
    image = np.array(image).astype(np.float32)
    C, H, W = image.shape
    patches, coords, H, W = extract_patches(image, patch_size, stride)
    vote_map = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
        
    with torch.no_grad():
        batch_patch_arr = np.array(patches)
        batch_patch_arr = z_score_normalize(batch_patch_arr)
        batch_patch_tensor = torch.from_numpy(batch_patch_arr).float().to(device)
        batch_pred = model(batch_patch_tensor)  # [B, num_classes, patch_size, patch_size]
        
    for b in range(batch_pred.shape[0]):
        pred = batch_pred[b]
        pred_label = torch.argmax(pred, dim=0).squeeze(0)  # [patch_size, patch_size]
        for cls in range(num_classes):
            mask = (pred_label == cls).float()
            vote_map[cls, coords[b][0]:coords[b][0]+patch_size, coords[b][1]:coords[b][1]+patch_size] += mask

    pred_map = torch.argmax(vote_map, dim=0).long()  # [H, W]
    
    if is_shown:
        plt.imshow(image[0], cmap='gray')
        plt.imshow(pred_map, alpha=0.5, cmap=cmap or 'jet')
        plt.tight_layout()
        plt.show()
    return pred_map.cpu().numpy()

def get_LV_center(pred_map):
    y_coords, x_coords = np.where(pred_map == 3) 
    
    if len(x_coords) == 0:
        center_x, center_y = None, None
    else:
        center_x = x_coords.mean()
        center_y = y_coords.mean()
    return center_x, center_y

def get_bbox(center_x, center_y, W, H, crop_w, crop_y):
    center_x, center_y, W, H, crop_w, crop_h = int(center_x), int(center_y), int(W), int(H), int(crop_w), int(crop_y)
    x1, y1, x2, y2 = -1, -1, -1, -1
    if center_x - crop_w//2 < 0:
        x1 = 0
        x2 = center_x + crop_w//2 + abs(center_x - crop_w//2)
    elif (center_x + crop_w/2) > W:
        x2 = W
        x1 = center_x - crop_w//2 - abs(crop_w//2-W+center_x)
    else:
        x1 = center_x - crop_w//2
        x2 = center_x + crop_w//2

    if center_y - crop_h//2 < 0:
        y1 = 0
        y2 = center_y + crop_h//2 + abs(center_y - crop_h//2)
    elif (center_y + crop_h//2) > H:
        y2 = H
        y1 = center_y - crop_h//2 - abs(crop_h//2-H+center_y)
    else:
        y1 = center_y - crop_h//2
        y2 = center_y + crop_h//2
    return int(x1), int(y1), int(x2), int(y2)

def standardize_resolution(image, spacing, target_spacing=(0.994, 0.994)):
    """
    image_np: shape = (W, H, S)  3D volume
    original_spacing: tuple of (sx, sy, sz)
    return: resampled image_np, new_spacing
    """
    image_array=np.transpose(image, (2, 1, 0))  ## H W S
    H, W, S = image_array.shape
    sx, sy, sz = spacing
    tx, ty = target_spacing

    scale_x = sx / tx
    scale_y = sy / ty

    new_H = int(round(H * scale_y))
    new_W = int(round(W * scale_x))

    resampled_slices = []
    for i in range(S):
        slice_i = image_array[:, :, i]

        # 对每一张slice进行resize
        resized = cv2.resize(
            slice_i,
            (new_W, new_H),  # 注意cv2顺序是 (width, height)
            interpolation=cv2.INTER_LINEAR  # 分割标签用 INTER_NEAREST
        )
        resampled_slices.append(resized)

    resampled_np = np.stack(resampled_slices, axis=-1)  # shape: (new_H, new_W, S)
    image_arr = np.transpose(resampled_np, (2, 1, 0))
    return image_arr, (tx, ty, sz)