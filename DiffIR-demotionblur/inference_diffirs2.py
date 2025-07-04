import argparse
import cv2
import glob
import numpy as np
import os
import torch

from DiffIR.archs.S2_arch import DiffIRS2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        '/home/ubuntu/data_sota_disk/model_space/diffIR/Deblurring-DiffIRS2.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='/home/ubuntu/data_sota_disk/dataset/diffIR_dblur/test_mini/input', help='input test image folder')
    parser.add_argument('--gt', type=str, default='/home/ubuntu/data_sota_disk/dataset/diffIR_dblur/test_mini/target', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/DiffIRS2', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    """
    type: DiffIRS2
  n_encoder_res: 5
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [3,5,6,6]
  num_refinement_blocks: 4
  heads: [1,2,4,8]
  ffn_expansion_factor: 2
  bias: False
  LayerNorm_type: WithBias
  n_denoise_res: 1
  linear_start: 0.1
  linear_end: 0.99
  timesteps: 4
  """
    model_param_dict = {
        'type': 'DiffIRS2',
        'n_encoder_res': 5,
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [3,5,6,6],
        'num_refinement_blocks': 4,
        'heads': [1,2,4,8],
        'ffn_expansion_factor': 2,
        'bias': False,
        'LayerNorm_type': 'WithBias',
        'n_denoise_res': 1,
        'linear_start': 0.1,
        'linear_end': 0.99,
        'timesteps': 4
    }
    model = DiffIRS2(n_encoder_res=5, inp_channels=3, out_channels=3, dim=48, num_blocks=[3,5,6,6], num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias', n_denoise_res=1, linear_start=0.1, linear_end=0.99, timesteps=4)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_DiffIRS2.png'), output)


if __name__ == '__main__':
    main()
