# ------------------------------------------------------------------------
# Test script for WaveMambaFlare with G/S-PSNR metrics
# Uses PairedImageDatasetWithMask/PairedImageDatasetRealTest for proper mask support
# ------------------------------------------------------------------------
import argparse
import copy
import logging
import os.path as osp
import torch
from collections import OrderedDict
from tqdm import tqdm

from basicsr.data import create_dataset, create_dataloader
from basicsr.metrics import calculate_psnr, calculate_ssim, calculate_g_psnr, calculate_s_psnr
from basicsr.utils import get_root_logger, imwrite, tensor2img


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
    args = parser.parse_args()
    return args


def main():
    args = parse_options()

    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    # Load option file
    import yaml
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    # Create result directory
    result_path = osp.join('results', opt['name'])
    os.makedirs(result_path, exist_ok=True)

    # Setup logger
    log_file = osp.join(result_path, f'test_{opt["name"]}.log')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Testing {opt["name"]}')

    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt.get('num_gpu', 1),
            dist=opt.get('dist', False),
            sampler=None,
            seed=opt.get('manual_seed', 0))
        test_loaders.append(test_loader)
        logger.info(f'Number of test images in {dataset_opt["name"]}: {len(test_set)}')

    # Build network
    from basicsr.archs import build_network
    network = build_network(opt['network_g'])
    network = network.cuda()

    # Load pretrained model
    load_path = opt['path']['pretrain_network_g']
    logger.info(f'Loading model from {load_path}')
    loaded = torch.load(load_path, map_location='cuda')
    if 'params' in loaded:
        # BasicSR format
        network.load_state_dict(loaded['params'], strict=True)
    else:
        network.load_state_dict(loaded, strict=True)

    network.eval()

    # Test each dataset
    for test_loader in test_loaders:
        dataset_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {dataset_name}...')

        # Initialize metrics
        metric_results = {k: 0 for k in ['psnr', 'ssim', 'g_psnr', 's_psnr']}
        cnt = 0

        pbar = tqdm(test_loader, unit='image')
        for idx, val_data in enumerate(pbar):
            lq = val_data['lq'].cuda()
            gt = val_data['gt'].cuda()
            mask = val_data.get('mask', None)
            if mask is not None:
                mask = mask.cuda()

            lq_path = val_data.get('lq_path', ['val'])[0]
            img_name = osp.splitext(osp.basename(lq_path))[0]

            # Forward
            with torch.no_grad():
                B, C, H, W = lq.shape
                if H * W < 8000 * 8000:
                    output, _ = network.test(lq)
                else:
                    output, _ = network.test_tile(lq)

            # Convert to numpy images
            sr_img = tensor2img([output.squeeze(0).float().cpu()])
            gt_img = tensor2img([gt.squeeze(0).float().cpu()])

            # Save image
            save_path = osp.join(result_path, f'{dataset_name}', f'{img_name}.png')
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            imwrite(sr_img, save_path)

            # Calculate metrics
            psnr_val = calculate_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False)
            ssim_val = calculate_ssim(sr_img, gt_img, crop_border=0, test_y_channel=False)

            g_psnr_val = 0
            s_psnr_val = 0
            if mask is not None:
                mask_img = tensor2img([mask.squeeze(0).float().cpu()])
                g_psnr_val = calculate_g_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False, mask=mask_img)
                s_psnr_val = calculate_s_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False, mask=mask_img)
            else:
                g_psnr_val = psnr_val
                s_psnr_val = psnr_val

            metric_results['psnr'] += psnr_val
            metric_results['ssim'] += ssim_val
            metric_results['g_psnr'] += g_psnr_val
            metric_results['s_psnr'] += s_psnr_val
            cnt += 1

            pbar.set_description(f'Test {img_name}: PSNR={psnr_val:.4f}, G-PSNR={g_psnr_val:.4f}')

        pbar.close()

        # Average metrics
        avg_metrics = {k: v / cnt for k, v in metric_results.items()}
        log_str = f'Validation {dataset_name}, '
        for k, v in avg_metrics.items():
            log_str += f'\t # {k}: {v:.4f}'
        logger.info(log_str)


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
