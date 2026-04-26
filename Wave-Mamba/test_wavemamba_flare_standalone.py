#!/usr/bin/env python
# ------------------------------------------------------------------------
# Standalone test script for WaveMambaFlare with G/S-PSNR metrics
# ------------------------------------------------------------------------
import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
import cv2
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, '/home/hjz/cv/Wave-Mamba Flare Removal/flare-removal/Wave-Mamba')

from basicsr.archs.wavemamba_flare_arch import WaveMambaFlare
from basicsr.utils.file_client import FileClient
from basicsr.utils.img_util import imfrombytes, img2tensor
from basicsr.utils.logger import get_root_logger


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find interested files."""
    for entry in os.scandir(dir_path):
        if not entry.name.startswith('.') and entry.is_file():
            if full_path:
                return_path = entry.path
            else:
                return_path = osp.relpath(entry.path, dir_path)
            if suffix is None or return_path.endswith(suffix):
                yield return_path
        elif recursive and entry.is_dir():
            yield from scandir(entry.path, suffix, recursive, full_path)


class PairedImageDatasetWithMask:
    """Dataset for synthetic test set with mask."""

    def __init__(self, opt):
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt.get('dataroot_mask', None)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # Build paths
        gt_paths = sorted(list(scandir(self.gt_folder, suffix='.png', full_path=False)))
        lq_paths = sorted(list(scandir(self.lq_folder, suffix='.png', full_path=False)))

        self.paths = []
        for gt_path in gt_paths:
            basename = osp.splitext(gt_path)[0]
            lq_path = f'{self.filename_tmpl.format(basename)}.png'
            if lq_path not in lq_paths:
                continue
            self.paths.append({
                'gt_path': osp.join(self.gt_folder, gt_path),
                'lq_path': osp.join(self.lq_folder, lq_path)
            })

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        mask = None
        if self.mask_folder:
            gt_basename = osp.basename(gt_path)
            mask_filename = f'mask_{gt_basename}'
            mask_path = osp.join(self.mask_folder, mask_filename)
            try:
                mask_bytes = self.file_client.get(mask_path, 'mask')
                mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0
            except Exception:
                pass

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        result = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

        if mask is not None:
            mask = mask[:, :, np.newaxis]
            mask_tensor = img2tensor([mask], bgr2rgb=False, float32=True)[0]
            result['mask'] = mask_tensor

        return result


class PairedImageDatasetRealTest:
    """Dataset for real test set with mask."""

    def __init__(self, opt):
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt.get('dataroot_mask', None)

        gt_paths = sorted(list(scandir(self.gt_folder, suffix='.png', full_path=False)))

        self.paths = []
        for gt_path in gt_paths:
            gt_basename = osp.splitext(gt_path)[0]  # e.g., gt_000018
            prefix = gt_basename.split('_')[0]  # 'gt'
            numeric_id = gt_basename.split('_')[1]  # '000018'
            lq_filename = f'input_{numeric_id}.png'
            self.paths.append({
                'gt_path': osp.join(self.gt_folder, gt_path),
                'lq_path': osp.join(self.lq_folder, lq_filename)
            })

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        mask = None
        if self.mask_folder:
            gt_basename = osp.basename(gt_path)
            gt_name = osp.splitext(gt_basename)[0]
            numeric_id = gt_name.split('_')[1]
            mask_filename = f'mask_{numeric_id}.png'
            mask_path = osp.join(self.mask_folder, mask_filename)
            try:
                mask_bytes = self.file_client.get(mask_path, 'mask')
                mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0
            except Exception:
                pass

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        result = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

        if mask is not None:
            mask = mask[:, :, np.newaxis]
            mask_tensor = img2tensor([mask], bgr2rgb=False, float32=True)[0]
            result['mask'] = mask_tensor

        return result


def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
    """Calculate PSNR."""
    if img1.ndim == 3 and img1.shape[2] == 3:
        if test_y_channel:
            img1 = (16 + 65.738 * img1[:, :, 0] + 129.057 * img1[:, :, 1] + 25.064 * img1[:, :, 2]) / 256.0
            img2 = (16 + 65.738 * img2[:, :, 0] + 129.057 * img2[:, :, 1] + 25.064 * img2[:, :, 2]) / 256.0
        else:
            img1 = img1[:, :, [2, 1, 0]]
            img2 = img2[:, :, [2, 1, 0]]
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0, test_y_channel=False):
    """Calculate SSIM using pyiqa."""
    import pyiqa
    ssim_metric = pyiqa.create_metric('ssim', device='cuda')
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ssim_metric(img1_tensor, img2_tensor).item()


def calculate_g_psnr(img1, img2, crop_border=0, test_y_channel=False, mask=None):
    """Calculate PSNR on glare/flare region using mask."""
    if mask is None:
        return calculate_psnr(img1, img2, crop_border, test_y_channel)

    if test_y_channel:
        img1 = (16 + 65.738 * img1[:, :, 0] + 129.057 * img1[:, :, 1] + 25.064 * img1[:, :, 2]) / 256.0
        img2 = (16 + 65.738 * img2[:, :, 0] + 129.057 * img2[:, :, 1] + 25.064 * img2[:, :, 2]) / 256.0

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        mask = mask[crop_border:-crop_border, crop_border:-crop_border]

    # Normalize mask if needed
    if mask.max() > 1.0:
        mask = mask / 255.0

    diff = (img1 - img2) ** 2
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    mse = np.sum(diff * mask) / (np.sum(mask) + 1e-8)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_s_psnr(img1, img2, crop_border=0, test_y_channel=False, mask=None):
    """Calculate PSNR on scene/non-flare region using inverse mask."""
    if mask is None:
        return calculate_psnr(img1, img2, crop_border, test_y_channel)

    # Normalize mask and invert for scene region
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = 1.0 - mask

    return calculate_g_psnr(img1, img2, crop_border, test_y_channel, mask)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch tensor to numpy image."""
    if not torch.is_tensor(tensor):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    img_np = tensor.numpy()
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        if rgb2bgr and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np.astype(out_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))

    # Load option file
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    result_path = osp.join('results', opt['name'])
    os.makedirs(result_path, exist_ok=True)

    logger = get_root_logger(logger_name='basicsr', log_level=20, log_file=osp.join(result_path, f'test_{opt["name"]}.log'))
    logger.info(f'Testing {opt["name"]}')

    # Create dataset
    dataset_opt = list(opt['datasets'].values())[0]
    if dataset_opt['type'] == 'PairedImageDatasetWithMask':
        test_set = PairedImageDatasetWithMask(dataset_opt)
    elif dataset_opt['type'] == 'PairedImageDatasetRealTest':
        test_set = PairedImageDatasetRealTest(dataset_opt)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_opt["type"]}')

    logger.info(f'Number of test images: {len(test_set)}')

    # Build network
    network = WaveMambaFlare(**opt['network_g'])
    network = network.cuda()
    network.eval()

    # Load model
    load_path = opt['path']['pretrain_network_g']
    logger.info(f'Loading model from {load_path}')
    loaded = torch.load(load_path, map_location='cuda')
    if 'params' in loaded:
        network.load_state_dict(loaded['params'], strict=True)
    else:
        network.load_state_dict(loaded, strict=True)

    # Initialize metrics
    metric_results = {k: 0 for k in ['psnr', 'ssim', 'g_psnr', 's_psnr']}
    cnt = 0

    pbar = tqdm(range(len(test_set)), unit='image')
    for idx in pbar:
        val_data = test_set[idx]
        lq = val_data['lq'].unsqueeze(0).cuda()
        gt = val_data['gt'].unsqueeze(0).cuda()
        mask = val_data.get('mask', None)
        if mask is not None:
            mask = mask.unsqueeze(0).cuda()

        img_name = osp.splitext(osp.basename(val_data['lq_path']))[0]

        # Forward
        with torch.no_grad():
            output = network.test(lq)

        # Convert to numpy
        sr_img = tensor2img(output.squeeze(0))
        gt_img = tensor2img(gt.squeeze(0))
        mask_np = None
        if mask is not None:
            mask_np = tensor2img(mask.squeeze(0), rgb2bgr=False)

        # Save image
        save_path = osp.join(result_path, dataset_opt['name'], f'{img_name}.png')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, sr_img)

        # Calculate metrics
        psnr_val = calculate_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False)
        ssim_val = calculate_ssim(sr_img, gt_img, crop_border=0, test_y_channel=False)
        g_psnr_val = calculate_g_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False, mask=mask_np)
        s_psnr_val = calculate_s_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False, mask=mask_np)

        metric_results['psnr'] += psnr_val
        metric_results['ssim'] += ssim_val
        metric_results['g_psnr'] += g_psnr_val
        metric_results['s_psnr'] += s_psnr_val
        cnt += 1

        pbar.set_description(f'Test {img_name}: PSNR={psnr_val:.4f}, G-PSNR={g_psnr_val:.4f}')

    pbar.close()

    # Average
    avg_metrics = {k: v / cnt for k, v in metric_results.items()}
    log_str = f'Validation {dataset_opt["name"]}, '
    for k, v in avg_metrics.items():
        log_str += f'\t # {k}: {v:.4f}'
    logger.info(log_str)
    print(log_str)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()
