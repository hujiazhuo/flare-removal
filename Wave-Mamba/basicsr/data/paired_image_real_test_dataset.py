# ------------------------------------------------------------------------
# Dedicated dataset for Real Test Set with mask for G/S-PSNR calculation
# Handles the different filename prefixes (gt_ vs input_) in Real Test Set
# ------------------------------------------------------------------------
from os import path as osp
import os
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor


class PairedImageDatasetRealTest(data.Dataset):
    """Paired image dataset for Real Test Set with mask support.

    Real Test Set has different filename conventions:
    - gt: gt_XXXXXX.png
    - input: input_XXXXXX.png
    - mask: mask_XXXXXX.png

    Args:
        opt (dict): Config for dataset. It contains:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_mask (str): Data root path for mask (optional).
            io_backend (dict): IO backend type.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealTest, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt.get('dataroot_mask', None)

        # Build paired paths manually for Real Test Set
        # Extract numeric ID from filenames and match by ID
        from basicsr.utils import scandir
        gt_paths = sorted(list(scandir(self.gt_folder)))
        lq_paths = sorted(list(scandir(self.lq_folder)))

        self.paths = []
        for gt_path in gt_paths:
            # gt_000018.png -> 000018
            basename, ext = osp.splitext(gt_path)
            prefix = basename.split('_')[0]  # 'gt'
            numeric_id = basename.split('_')[1]  # '000018'

            # Find corresponding lq file: input_000018.png
            lq_filename = f'input_{numeric_id}{ext}'
            if lq_filename not in lq_paths:
                raise AssertionError(f'{lq_filename} is not in lq_paths.')

            self.paths.append({
                'gt_path': osp.join(self.gt_folder, gt_path),
                'lq_path': osp.join(self.lq_folder, lq_filename)
            })

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt image
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        # Load lq image
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # Load mask if available
        mask = None
        if self.mask_folder is not None:
            # Build mask path from gt_path numeric ID
            # gt_000018.png -> mask_000018.png
            gt_basename = osp.basename(gt_path)  # gt_000018.png
            gt_name = osp.splitext(gt_basename)[0]  # gt_000018
            numeric_id = gt_name.split('_')[1]  # 000018
            mask_filename = f'mask_{numeric_id}.png'
            mask_path = osp.join(self.mask_folder, mask_filename)
            try:
                import cv2
                mask_bytes = self.file_client.get(mask_path, 'mask')
                mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0  # Normalize to [0, 1]
            except Exception:
                # Mask not found, use None
                pass

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        result = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

        if mask is not None:
            # Add channel dimension for img2tensor (HWC format)
            mask = mask[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
            mask_tensor = img2tensor([mask], bgr2rgb=False, float32=True)[0]
            result['mask'] = mask_tensor

        return result

    def __len__(self):
        return len(self.paths)