"""
Outcome classification of phase-retrieval samples (PC-DAPS paper, Sec. 5 and Table 2).

Every sample is compared, channel by channel, with the ground truth and with its 180-degree rotation (its
measurement-equivalent twin), maximizing the PSNR over +-1-pixel circular shifts to absorb the translation
ambiguity. A channel is assigned to an orientation only when it is more than `margin_db` dB closer to it:
    faithful : all channels match the ground truth,
    flipped  : every channel is assigned and at least one matches the rotation
               (upside-down, or color-inverted when the channels disagree),
    failed   : the remainder.
An image is unrecoverable when none of its samples (independent runs or particles of a bag) is faithful.

Usage on saved samples ({image:05d}_run{run:04d}[_p{particle:02d}].png):
    python pr_outcomes.py --samples results/pixel/ffhq/phase_retrieval/samples --gt dataset/test-ffhq --end_id 100
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

LABELS = ('faithful', 'flipped', 'failed')


def channel_psnr(x, ref):
    """Per-channel PSNR [N, C] of images in [0, 1]."""
    mse = (x - ref).pow(2).mean((-2, -1)).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


@torch.no_grad()
def classify(samples, gt, max_shift=1, margin_db=1.0):
    """
    Args:
        samples (torch.Tensor): Samples [N, C, H, W] in [-1, 1].
        gt (torch.Tensor): Ground truth [N, C, H, W] in [-1, 1].
        max_shift (int): Largest circular shift (pixels) searched in each direction.
        margin_db (float): PSNR margin required to assign a channel to an orientation.

    Returns:
        np.ndarray: Labels [N] as indices into LABELS.
    """
    x = (samples * 0.5 + 0.5).clamp(0, 1)
    g = (gt.to(x.device) * 0.5 + 0.5).clamp(0, 1)
    shifts = [(dy, dx) for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)]
    best = []
    for ref in (g, g.flip(-2, -1)):
        psnr = torch.stack([channel_psnr(x, torch.roll(ref, shift, dims=(-2, -1))) for shift in shifts])
        best.append(psnr.max(0).values)
    upright = best[0] > best[1] + margin_db
    rotated = best[1] > best[0] + margin_db
    faithful = upright.all(1)
    flipped = ~faithful & (upright | rotated).all(1)
    return np.where(faithful.cpu().numpy(), 0, np.where(flipped.cpu().numpy(), 1, 2))


def summarize(samples, gt, **kwargs):
    """
    Args:
        samples (torch.Tensor): Samples [S, N, C, H, W] (S runs or particles per image) in [-1, 1].
        gt (torch.Tensor): Ground truth [N, C, H, W] in [-1, 1].

    Returns:
        dict: Outcome frequencies over all samples, rate of unrecoverable images and per-image labels.
    """
    labels = np.stack([classify(s, gt, **kwargs) for s in samples])     # [S, N]
    result = {name: float((labels == i).mean()) for i, name in enumerate(LABELS)}
    result['unrecoverable'] = float((labels != 0).all(0).mean())
    result['samples_per_image'] = int(labels.shape[0])
    result['labels'] = [[LABELS[label] for label in labels[:, n]] for n in range(labels.shape[1])]
    return result


def format_summary(result):
    return ('phase-retrieval outcomes ({} samples per image): faithful {:.1%} | upside-down / color-inverted {:.1%} | '
            'failed {:.1%} | unrecoverable images {:.1%}').format(result['samples_per_image'], result['faithful'],
                                                                  result['flipped'], result['failed'], result['unrecoverable'])


def load_image(path, resolution):
    img = TF.to_tensor(Image.open(path).convert('RGB'))
    img = TF.center_crop(TF.resize(img, resolution, antialias=True), resolution)
    return img * 2 - 1


def main():
    parser = argparse.ArgumentParser(description='Classify saved phase-retrieval samples as faithful / flipped / failed.')
    parser.add_argument('--samples', required=True, help='directory with {image:05d}_run{run:04d}[_p{particle:02d}].png')
    parser.add_argument('--gt', required=True, help='directory of the ground-truth images used for sampling')
    parser.add_argument('--start_id', type=int, default=0, help='data.start_id used for sampling')
    parser.add_argument('--end_id', type=int, default=None, help='data.end_id used for sampling')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--margin_db', type=float, default=1.0)
    parser.add_argument('--max_shift', type=int, default=1)
    parser.add_argument('--output', default=None, help='optional json output path')
    args = parser.parse_args()

    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    gt_files = sorted(file for ext in extensions for file in Path(args.gt).rglob(ext))[args.start_id:args.end_id]
    samples = defaultdict(list)
    for path in sorted(Path(args.samples).glob('*.png')):
        match = re.fullmatch(r'(\d+)_run\d+(_p\d+)?\.png', path.name)
        if match:
            samples[int(match.group(1))].append(path)
    indices = sorted(samples)
    counts = {len(samples[i]) for i in indices}
    if len(counts) != 1:
        raise ValueError(f'images have different numbers of samples: {sorted(counts)}')

    gt = torch.stack([load_image(gt_files[i], args.resolution) for i in indices])
    stack = torch.stack([torch.stack([load_image(p, args.resolution) for p in samples[i]]) for i in indices], dim=1)
    result = summarize(stack, gt, max_shift=args.max_shift, margin_db=args.margin_db)
    print(format_summary(result))
    if args.output:
        json.dump(result, open(args.output, 'w'), indent=4)


if __name__ == '__main__':
    main()
