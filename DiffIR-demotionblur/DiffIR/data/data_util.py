import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from DiffIR.data.transforms import mod_crop
from DiffIR.utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs


def generate_frame_indices(crt_idx,
                           max_frame_num,
                           num_frames,
                           padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths

def paired_DP_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [inputL_folder, inputR_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [inputL_folder, inputR_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 2 with [inputL_key, inputR_key, gt_key]. '
        f'But got {len(keys)}')
    inputL_folder, inputR_folder, gt_folder = folders
    inputL_key, inputR_key, gt_key = keys

    inputL_paths = list(scandir(inputL_folder))
    inputR_paths = list(scandir(inputR_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(inputL_paths) == len(inputR_paths) == len(gt_paths), (
        f'{inputL_key} and {inputR_key} and {gt_key} datasets have different number of images: '
        f'{len(inputL_paths)}, {len(inputR_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        inputL_path = inputL_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputL_path))
        inputL_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputL_path = osp.join(inputL_folder, inputL_name)
        assert inputL_name in inputL_paths, (f'{inputL_name} is not in '
                                           f'{inputL_key}_paths.')
        inputR_path = inputR_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputR_path))
        inputR_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputR_path = osp.join(inputR_folder, inputR_name)
        assert inputR_name in inputR_paths, (f'{inputR_name} is not in '
                                           f'{inputR_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{inputL_key}_path', inputL_path),
                  (f'{inputR_key}_path', inputR_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x


def triple_paths_from_folder(folders, keys, filename_tmpl):
    """Generate triple paths from folders for HDR processing.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [ldr_folder, hdr_folder, dgain_info_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['ldr', 'hdr', 'dgain'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the ldr folder (input folder).

    Returns:
        list[dict]: Returned path list. Each item contains paths for ldr, hdr, and dgain.
        
    Example:
        folders = ['/path/to/ldr', '/path/to/hdr', '/path/to/dgain']
        keys = ['ldr', 'hdr', 'dgain']
        filename_tmpl = '{}'
        
        Returns:
        [
            {'ldr_path': '/path/to/ldr/img001.jpg', 
             'hdr_path': '/path/to/hdr/img001.hdr', 
             'dgain_path': '/path/to/dgain/img001.txt'},
            ...
        ]
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [ldr_folder, hdr_folder, dgain_info_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [ldr_key, hdr_key, dgain_key]. '
        f'But got {len(keys)}')
    
    ldr_folder, hdr_folder, dgain_folder = folders
    ldr_key, hdr_key, dgain_key = keys

    # 获取所有文件夹中的文件列表
    ldr_paths = list(scandir(ldr_folder))
    hdr_paths = list(scandir(hdr_folder))
    dgain_paths = list(scandir(dgain_folder))
    
    # 检查文件数量是否一致
    assert len(ldr_paths) == len(hdr_paths) == len(dgain_paths), (
        f'{ldr_key}, {hdr_key} and {dgain_key} datasets have different number of files: '
        f'{len(ldr_paths)}, {len(hdr_paths)}, {len(dgain_paths)}.')
    
    paths = []
    for idx in range(len(ldr_paths)):
        # 以LDR文件作为基准文件名
        ldr_path = ldr_paths[idx]
        basename, ext = osp.splitext(osp.basename(ldr_path))
        
        # 构建对应的HDR和dgain文件路径
        hdr_name = f'{filename_tmpl.format(basename)}'
        dgain_name = f'{filename_tmpl.format(basename)}'
        
        # 在对应文件夹中查找匹配的文件
        hdr_path = None
        dgain_path = None
        
        # 查找HDR文件（可能有不同扩展名：.hdr, .exr, .tiff等）
        for hdr_file in hdr_paths:
            hdr_basename, hdr_ext = osp.splitext(osp.basename(hdr_file))
            if hdr_basename == basename:
                hdr_path = osp.join(hdr_folder, hdr_file)
                break
        
        # 查找dgain文件（可能有不同扩展名：.txt, .json, .xml等）
        for dgain_file in dgain_paths:
            dgain_basename, dgain_ext = osp.splitext(osp.basename(dgain_file))
            if dgain_basename == basename:
                dgain_path = osp.join(dgain_folder, dgain_file)
                break
        
        # 确保找到了所有对应的文件
        assert hdr_path is not None, (
            f'Cannot find corresponding HDR file for {basename} in {hdr_folder}')
        assert dgain_path is not None, (
            f'Cannot find corresponding dgain file for {basename} in {dgain_folder}')
        
        # 构建完整路径
        ldr_full_path = osp.join(ldr_folder, ldr_path)
        
        paths.append(
            dict([
                (f'{ldr_key}_path', ldr_full_path),
                (f'{hdr_key}_path', hdr_path),
                (f'{dgain_key}_path', dgain_path)
            ]))
    
    return paths


def triple_paths_from_folder_flexible(folders, keys, filename_tmpl, extensions=None):
    """Generate triple paths from folders with flexible file extension matching.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [ldr_folder, hdr_folder, dgain_info_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['ldr', 'hdr', 'dgain'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension.
        extensions (list[list[str]], optional): Expected file extensions for each folder.
            e.g., [['jpg', 'png'], ['hdr', 'exr'], ['txt', 'json']]
            If None, will match any extension.

    Returns:
        list[dict]: Returned path list. Each item contains paths for ldr, hdr, and dgain.
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [ldr_folder, hdr_folder, dgain_info_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [ldr_key, hdr_key, dgain_key]. '
        f'But got {len(keys)}')
    
    if extensions is not None:
        assert len(extensions) == 3, (
            'The len of extensions should be 3 to match folders. '
            f'But got {len(extensions)}')
    
    ldr_folder, hdr_folder, dgain_folder = folders
    ldr_key, hdr_key, dgain_key = keys

    # 获取所有文件夹中的文件列表
    ldr_paths = list(scandir(ldr_folder))
    hdr_paths = list(scandir(hdr_folder))
    dgain_paths = list(scandir(dgain_folder))
    
    # 根据扩展名过滤文件（如果指定了extensions）
    if extensions:
        ldr_exts, hdr_exts, dgain_exts = extensions
        ldr_paths = [f for f in ldr_paths if any(f.lower().endswith(f'.{ext}') for ext in ldr_exts)]
        hdr_paths = [f for f in hdr_paths if any(f.lower().endswith(f'.{ext}') for ext in hdr_exts)]
        dgain_paths = [f for f in dgain_paths if any(f.lower().endswith(f'.{ext}') for ext in dgain_exts)]
    
    # 创建文件名到路径的映射
    def create_basename_map(file_paths, folder):
        basename_map = {}
        for file_path in file_paths:
            basename = osp.splitext(osp.basename(file_path))[0]
            basename_map[basename] = osp.join(folder, file_path)
        return basename_map
    
    ldr_map = create_basename_map(ldr_paths, ldr_folder)
    hdr_map = create_basename_map(hdr_paths, hdr_folder)
    dgain_map = create_basename_map(dgain_paths, dgain_folder)
    
    # 找到所有三个文件夹都有的文件名
    common_basenames = set(ldr_map.keys()) & set(hdr_map.keys()) & set(dgain_map.keys())
    
    if not common_basenames:
        raise ValueError(
            f'No common files found across all three folders: '
            f'{ldr_folder}, {hdr_folder}, {dgain_folder}')
    
    print(f"Found {len(common_basenames)} common files across all folders")
    
    paths = []
    for basename in sorted(common_basenames):
        paths.append(
            dict([
                (f'{ldr_key}_path', ldr_map[basename]),
                (f'{hdr_key}_path', hdr_map[basename]),
                (f'{dgain_key}_path', dgain_map[basename])
            ]))
    
    return paths
