import argparse
import cv2
import glob
import numpy as np
import os
import torch
from DiffIR.data.common_function import load_hdr,load_ldr_file,extract_info_from_dgain_prompt,log_tone_mapping,inverse_custom_tone_mapping,inverse_log_tone_mapping
from DiffIR.archs.S1_arch import DiffIRS1
from metrics.metrics_calculate import calculate_psnr,calculate_ssim
from metrics.pu21_metrics import pu21_metric
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


def load_model_weights(model, model_path, device, use_ema=None, no_ema=False):
    """
    加载模型权重，优先使用EMA权重
    
    Args:
        model: 模型实例
        model_path: 模型权重文件路径
        device: 计算设备
        use_ema: 强制使用EMA权重
        no_ema: 强制跳过EMA权重
    
    Returns:
        model: 加载权重后的模型
    """
    print(f"Loading model from: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 打印checkpoint中的键，用于调试
    print("Available keys in checkpoint:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # 根据用户偏好决定加载顺序
    if no_ema:
        print("用户指定跳过EMA权重，直接使用普通权重")
        use_ema_first = False
    elif use_ema:
        print("用户指定优先使用EMA权重")
        use_ema_first = True
    else:
        print("自动检测：优先使用EMA权重（如果可用）")
        use_ema_first = True
    
    # 尝试加载权重
    if use_ema_first and 'params_ema' in checkpoint:
        print("Found EMA weights, loading params_ema...")
        try:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
            print("✅ Successfully loaded EMA weights")
            
            # 比较EMA和普通权重的差异（如果两者都存在）
            if 'params' in checkpoint:
                compare_weights(checkpoint['params'], checkpoint['params_ema'])
            
            return model
        except Exception as e:
            print(f"❌ Failed to load EMA weights: {e}")
            if use_ema:
                raise ValueError("User requested EMA weights but failed to load")
            print("Falling back to regular weights...")
    
    # 如果没有EMA权重或EMA权重加载失败，使用普通权重
    if 'params' in checkpoint:
        print("Loading regular weights (params)...")
        try:
            model.load_state_dict(checkpoint['params'], strict=True)
            print("✅ Successfully loaded regular weights")
            return model
        except Exception as e:
            print(f"❌ Failed to load regular weights: {e}")
    
    # 如果上述都失败，尝试直接加载（某些情况下checkpoint就是state_dict）
    try:
        print("Trying to load checkpoint directly as state_dict...")
        model.load_state_dict(checkpoint, strict=True)
        print("✅ Successfully loaded weights directly")
        return model
    except Exception as e:
        print(f"❌ Failed to load weights directly: {e}")
        raise ValueError("Cannot load model weights from the checkpoint")


def compare_weights(params, params_ema, sample_keys=5):
    """
    比较普通权重和EMA权重的差异
    
    Args:
        params: 普通权重字典
        params_ema: EMA权重字典
        sample_keys: 采样显示的参数数量
    """
    print("\n📊 Comparing regular weights vs EMA weights:")
    
    common_keys = set(params.keys()) & set(params_ema.keys())
    if not common_keys:
        print("  No common keys found between params and params_ema")
        return
    
    print(f"  Common parameters: {len(common_keys)}")
    
    # 计算整体差异
    total_diff = 0
    total_norm = 0
    
    sampled_keys = list(common_keys)[:sample_keys]
    
    for key in sampled_keys:
        param = params[key]
        param_ema = params_ema[key]
        
        if param.shape != param_ema.shape:
            print(f"  ⚠️  Shape mismatch for {key}: {param.shape} vs {param_ema.shape}")
            continue
            
        diff = torch.norm(param - param_ema).item()
        norm = torch.norm(param).item()
        
        if norm > 0:
            relative_diff = diff / norm
            print(f"  {key}: relative diff = {relative_diff:.6f}")
            total_diff += diff
            total_norm += norm
    
    if total_norm > 0:
        overall_relative_diff = total_diff / total_norm
        print(f"  📈 Overall relative difference: {overall_relative_diff:.6f}")
        
        if overall_relative_diff < 0.01:
            print("  ✨ EMA and regular weights are very similar")
        elif overall_relative_diff < 0.1:
            print("  📝 EMA weights show moderate differences")
        else:
            print("  🔄 EMA weights show significant differences")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        '/home/ubuntu/data_sota_disk/model_space/diffIR/net_g_280000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/val_hdr/jpg', help='input test image folder')
    parser.add_argument('--dgain_folder', type=str, default='/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/val_hdr/dgain_info', help='input test image folder')
    parser.add_argument('--gt', type=str, default='/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/val_hdr/hdr', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/DiffIRS1_hdr_280000', help='output folder')
    parser.add_argument('--use_ema', action='store_true', help='强制使用EMA权重（如果可用）')
    parser.add_argument('--no_ema', action='store_true', help='强制使用普通权重，跳过EMA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # set up model
    model_param_dict = {
        'type': 'DiffIRS1',
        'n_encoder_res': 5,
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [3,5,6,6],
        'num_refinement_blocks': 4,
        'heads': [1,2,4,8],
        'ffn_expansion_factor': 2,
        'bias': False,
        'LayerNorm_type': 'WithBias'
    }
    
    model = DiffIRS1(
        n_encoder_res=5, 
        inp_channels=3, 
        out_channels=3, 
        dim=48, 
        num_blocks=[3,5,6,6], 
        num_refinement_blocks=4, 
        heads=[1,2,4,8], 
        ffn_expansion_factor=2, 
        bias=False, 
        LayerNorm_type='WithBias'
    )
    
    # 加载模型权重
    model = load_model_weights(model, args.model_path, device, args.use_ema, args.no_ema)
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded successfully and moved to {device}")

    os.makedirs(args.output, exist_ok=True)
    total_psnr = 0
    total_ssim = 0
    total_num = 0
    hdr_max = 1000
    I_peak = 4000
    total_psnr_pu21 = 0
    total_ssim_pu21 = 0
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print(f'Testing {idx+1}: {imgname}')
        
        # read image
        img = load_ldr_file(path)
        dgain_path = os.path.join(args.dgain_folder, f'{imgname}.txt')
        with open(dgain_path, 'r') as file:
            prompt = file.read()
        dgain_info = extract_info_from_dgain_prompt(prompt)
        dgain = dgain_info['dgain']
        gamma = dgain_info['gamma']
        

        img =inverse_custom_tone_mapping(img, dgain, gamma,max_value=1000)

        img = log_tone_mapping(img,hdr_max=1000)

        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        
        gt_path = os.path.join(args.gt, f'{imgname}.hdr')
        gt = load_hdr(gt_path)
        linear_gt = gt.copy()
        linear_gt_metric = linear_gt / hdr_max *I_peak
        gt = log_tone_mapping(gt, hdr_max=1000)

        gt_linear = inverse_log_tone_mapping(gt, hdr_max=1000)
        gt_linear_metrics = gt_linear / hdr_max * I_peak
        gt_bgr = gt.copy()
        #gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        gt = torch.from_numpy(np.transpose(gt[:, :, [2, 1, 0]], (2, 0, 1))).float()
        gt = gt.unsqueeze(0).to(device)
        #gt = log_tone_mapping(gt, hdr_max=1000)
        
        # inference
        try:
            with torch.no_grad():
                output = model(img, gt)
        except Exception as error:
            print(f'Error processing {imgname}: {error}')
            continue
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output_linear = inverse_log_tone_mapping(output, hdr_max=1000)
            output_linear_metrics = output_linear / hdr_max * I_peak
            #peak_signal_noise_ratio()
            current_psnr = calculate_psnr(output, gt_bgr)
            current_ssim = calculate_ssim(output, gt_bgr)

            psnr_pu21_recover = pu21_metric(output_linear_metrics, linear_gt_metric,metric='PSNR')
            ssim_pu21_recover = pu21_metric(output_linear_metrics, linear_gt_metric,metric='SSIM')
            output = (output * 255.0).round().astype(np.uint8)
            output_path = os.path.join(args.output, f'{imgname}_DiffIRS1_hdr.png')
            cv2.imwrite(output_path, output)
            print(f'Saved: {output_path}')
            print(f'PSNR: {current_psnr}, SSIM: {current_ssim}')
            print(f'PSNR_PU21: {psnr_pu21_recover}, SSIM_PU21: {ssim_pu21_recover}')
            total_psnr += current_psnr
            total_ssim += current_ssim
            total_psnr_pu21 += psnr_pu21_recover
            total_ssim_pu21 += ssim_pu21_recover
            total_num += 1
    print(f'Total PSNR: {total_psnr/total_num}, Total SSIM: {total_ssim/total_num}')
    print(f'Total PSNR_PU21: {total_psnr_pu21/total_num}, Total SSIM_PU21: {total_ssim_pu21/total_num}')
    print("Inference completed!")


def check_model_weights(model_path):
    """
    快速检查模型权重文件中包含的键
    
    Args:
        model_path: 模型权重文件路径
    """
    print(f"🔍 Checking model weights in: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("\nAvailable keys in checkpoint:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  ✅ {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  ✅ {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  ✅ {key}: {type(checkpoint[key])}")
        
        # 特别检查EMA权重
        if 'params_ema' in checkpoint:
            print(f"\n🎯 EMA weights found! (params_ema)")
            print(f"   EMA权重包含 {len(checkpoint['params_ema'])} 个参数")
        else:
            print(f"\n⚠️  No EMA weights found (params_ema not present)")
            
        if 'params' in checkpoint:
            print(f"🎯 Regular weights found! (params)")
            print(f"   普通权重包含 {len(checkpoint['params'])} 个参数")
        else:
            print(f"⚠️  No regular weights found (params not present)")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


if __name__ == '__main__':
    import sys
    
    # 如果命令行包含 --check-weights，只检查权重不运行推理
    if '--check-weights' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, 
                          default='/home/ubuntu/data_sota_disk/model_space/diffIR/Deblurring-DiffIRS1.pth')
        args = parser.parse_args()
        check_model_weights(args.model_path)
    else:
        main()
