import cv2
import random
import numpy as np

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqLs, list):
        img_lqLs = [img_lqLs]
    if not isinstance(img_lqRs, list):
        img_lqRs = [img_lqRs]

    h_lq, w_lq, _ = img_lqLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqLs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqLs
    ]

    img_lqRs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqRs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqLs) == 1:
        img_lqLs = img_lqLs[0]
    if len(img_lqRs) == 1:
        img_lqRs = img_lqRs[0]
    return img_lqLs, img_lqRs, img_gts


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def padding_triple(img_gt, img_lq, img_gt_recover, gt_size):
    """对三个图像进行padding
    
    Args:
        img_gt (ndarray): GT图像
        img_lq (ndarray): LQ图像  
        img_gt_recover (ndarray): GT_RECOVER图像
        gt_size (int): 目标尺寸
        
    Returns:
        tuple: padding后的三个图像
    """
    h, w = img_lq.shape[:2]
    pad_h = max(0, gt_size - h)
    pad_w = max(0, gt_size - w)
    if pad_h > 0 or pad_w > 0:
        img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        img_gt_recover = cv2.copyMakeBorder(img_gt_recover, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return img_gt, img_lq, img_gt_recover

def paired_random_crop_triple(img_gt, img_lq, img_gt_recover, gt_size, scale, gt_path):
    """对三个图像进行配对的随机裁剪
    
    Args:
        img_gt (ndarray): GT图像
        img_lq (ndarray): LQ图像  
        img_gt_recover (ndarray): GT_RECOVER图像
        gt_size (int): GT图像的裁剪尺寸
        scale (int): 缩放因子
        gt_path (str): GT图像路径（用于错误信息）
        
    Returns:
        tuple: 裁剪后的三个图像
    """
    h_lq, w_lq, _ = img_lq.shape
    h_gt, w_gt, _ = img_gt.shape
    h_recover, w_recover, _ = img_gt_recover.shape
    
    # 所有图像应该有相同的尺寸
    assert h_lq == h_gt == h_recover and w_lq == w_gt == w_recover, \
        f'GT and LQ and GT_RECOVER have different sizes: {(h_gt, w_gt)} vs {(h_lq, w_lq)} vs {(h_recover, w_recover)}'
    
    lq_size = gt_size // scale
    if h_lq != h_gt or w_lq != w_gt:
        raise ValueError(f'图像尺寸不匹配 {gt_path}: {(h_lq, w_lq)} vs {(h_gt, w_gt)} vs {(h_recover, w_recover)}')
    if h_lq < lq_size or w_lq < lq_size:
        raise ValueError(f'LQ图像尺寸太小 {gt_path}: ({h_lq}, {w_lq}) < {lq_size}')

    # 随机选择裁剪起始点
    top = random.randint(0, h_lq - lq_size)
    left = random.randint(0, w_lq - lq_size)
    
    # 对所有图像应用相同的裁剪
    img_lq = img_lq[top:top + lq_size, left:left + lq_size, ...]
    h_gt_size, w_gt_size = int(lq_size * scale), int(lq_size * scale)
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gt = img_gt[top_gt:top_gt + h_gt_size, left_gt:left_gt + w_gt_size, ...]
    img_gt_recover = img_gt_recover[top_gt:top_gt + h_gt_size, left_gt:left_gt + w_gt_size, ...]
    
    return img_gt, img_lq, img_gt_recover

def random_augmentation_triple(img_gt, img_lq, img_gt_recover):
    """对三个图像进行相同的随机增强
    
    Args:
        img_gt (ndarray): GT图像
        img_lq (ndarray): LQ图像  
        img_gt_recover (ndarray): GT_RECOVER图像
        
    Returns:
        tuple: 增强后的三个图像
    """
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hflip:
        img_gt = cv2.flip(img_gt, 1)
        img_lq = cv2.flip(img_lq, 1)
        img_gt_recover = cv2.flip(img_gt_recover, 1)
    if vflip:
        img_gt = cv2.flip(img_gt, 0)
        img_lq = cv2.flip(img_lq, 0)
        img_gt_recover = cv2.flip(img_gt_recover, 0)
    if rot90:
        img_gt = img_gt.transpose(1, 0, 2)
        img_lq = img_lq.transpose(1, 0, 2)
        img_gt_recover = img_gt_recover.transpose(1, 0, 2)
        
    return img_gt, img_lq, img_gt_recover
