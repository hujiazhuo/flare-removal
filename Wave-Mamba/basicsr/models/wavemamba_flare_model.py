# ------------------------------------------------------------------------
# Wave-Mamba Flare Model
# Model class for training WaveMambaFlare architecture
# ------------------------------------------------------------------------

from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
from .cal_ssim import SSIM
from torch import nn
import sys


@MODEL_REGISTRY.register()
class WaveMambaFlareModel(BaseModel):
    """
    Model class for WaveMambaFlare (Flare Removal).

    Key differences from base model:
    - Network outputs (clean_image, flare_predicted) instead of single restoration
    - Loss: L1(flare_pred, gt_flare) where gt_flare = input - gt_clean
    - Additional perceptual loss on clean image
    """

    def __init__(self, opt):
        super().__init__(opt)

        # Build network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.ssim = SSIM().cuda()
        self.l1 = nn.L1Loss().cuda()

        # Metrics for validation
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # Load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # Build loss functions
        if train_opt.get('flare_l1_opt'):
            self.cri_flare_l1 = build_loss(train_opt['flare_l1_opt']).to(self.device)
        else:
            self.cri_flare_l1 = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # Set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # Define optimizer
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)  # input with flare
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)  # clean ground truth

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()

        # Forward pass: returns (clean_image, flare_predicted)
        self.clean_pred, self.flare_pred = self.net_g(self.lq)

        # Compute gt_flare = input - gt_clean
        gt_flare = self.lq - self.gt

        l_g_total = 0
        loss_dict = OrderedDict()

        # 1. Flare prediction loss (L1 on flare component)
        if self.cri_flare_l1 is not None:
            l_flare = self.cri_flare_l1(self.flare_pred, gt_flare)
            l_g_total += l_flare
            loss_dict['l_flare'] = l_flare
        else:
            l_flare_l1 = self.l1(self.flare_pred, gt_flare)
            l_g_total += l_flare_l1
            loss_dict['l_flare_l1'] = l_flare_l1

        # 2. Clean image pixel loss (if configured)
        if self.cri_pix is not None:
            l_pix = self.cri_pix(self.clean_pred, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 3. Perceptual loss on clean image
        if self.cri_perceptual is not None:
            l_perc, _ = self.cri_perceptual(self.clean_pred, self.gt)
            if l_perc is not None:
                l_g_total += l_perc
                loss_dict['l_perc'] = l_perc

        # 4. FFT loss on clean image (if configured)
        if self.cri_fft is not None:
            l_fft = self.cri_fft(self.clean_pred, self.gt)
            l_g_total += l_fft
            loss_dict['l_fft'] = l_fft

        l_g_total.mean().backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        if h * w < min_size:
            # Forward returns only clean image for output
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
                           save_img, save_as_dir=None):
        dataset_name = 'Flare7K++'
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        for idx, val_data in enumerate(dataloader):
            img_name = f'val_{idx:04d}'
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # Free memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # Calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # Calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            # Update best metric and save model
            key_metric = self.opt['val'].get('key_metric')
            if key_metric is not None:
                to_update = self._update_best_metric_result(dataset_name, key_metric,
                                                            self.metric_results[key_metric], current_iter)
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
            else:
                # Update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]  # input with flare
        out_dict['clean'] = self.clean_pred.detach().cpu()[:vis_samples]  # predicted clean
        out_dict['flare'] = self.flare_pred.detach().cpu()[:vis_samples]  # predicted flare
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        self.save_training_state(epoch, current_iter)
