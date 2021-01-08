import torch.nn.functional as F
import cv2
import numpy as np
from datasets.burstsr_dataset import BurstSRDataset
from torch.utils.data.dataloader import DataLoader
from utils.metrics import AlignedPSNR
from utils.postprocessing_functions import BurstSRPostProcess
from utils.data_format_utils import convert_dict
from pwcnet.pwcnet import PWCNet


def main():
    dataset = BurstSRDataset(root='/media/goutam/T7/BurstSRChallenge',
                             split='val', burst_size=3, crop_sz=56, random_flip=False)

    data_loader = DataLoader(dataset, batch_size=2)

    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='/home/goutam/data/train_ws/pretrained_networks/pwcnet-network-default.pth')
    alignment_net = alignment_net.to('cuda')

    psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)

    postprocess_fn = BurstSRPostProcess(return_np=True)

    for d in data_loader:
        burst, frame_gt, meta_info_burst, meta_info_gt = d

        # Simple upsapling
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')

        score = psnr_fn(burst_rgb.cuda(), frame_gt.cuda(), burst.cuda())

        # Visualize input, ground truth
        meta_info_gt = convert_dict(meta_info_gt, burst.shape[0])

        pred_0 = postprocess_fn.process(burst_rgb[0], meta_info_gt[0])
        gt_0 = postprocess_fn.process(frame_gt[0], meta_info_gt[0])

        pred_0 = cv2.cvtColor(pred_0, cv2.COLOR_RGB2BGR)
        gt_0 = cv2.cvtColor(gt_0, cv2.COLOR_RGB2BGR)

        cv2.imshow('Pred', pred_0)
        cv2.imshow('GT', gt_0)

        input_key = cv2.waitKey(0)
        if input_key == ord('q'):
            return


if __name__ == '__main__':
    main()
