# ------------------------------------------------------------------------
# Modified from PairedImageDataset to support mask for G/S-PSNR calculation
# ------------------------------------------------------------------------
from os import path as osp
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor


class PairedImageDatasetWithMask(data.Dataset):
    """Paired image dataset with mask for flare removal evaluation.

    Supports G-PSNR (PSNR on glare/flare region) and S-PSNR (PSNR on scene region).

    Args:
        opt (dict): Config for dataset. It contains:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_mask (str): Data root path for mask (optional).
            io_backend (dict): IO backend type.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetWithMask, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt.get('dataroot_mask', None)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

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
            # Build mask path from lq_path
            # e.g., 000000.png -> mask_000000.png
            lq_filename = osp.basename(lq_path)
            mask_filename = 'mask_' + lq_filename
            mask_path = osp.join(self.mask_folder, mask_filename)
            try:
                mask_bytes = self.file_client.get(mask_path, 'mask')
                import cv2
                import numpy as np
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
