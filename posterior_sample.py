import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, Evaluator
import pr_outcomes
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import setproctitle
from PIL import Image
import numpy as np
import imageio

import os


def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def apply_task_pcdaps_config(args):
    """
        PC-DAPS: merge the task's `pcdaps_config` (e.g. configs/task/hdr.yaml) into the sampler config; options set
        on the command line (sampler.pcdaps_config.<key>=...) take precedence
    """
    if args.sampler.get('name') != 'pcdaps':
        return
    options = dict(args.task[args.task_group].get('pcdaps_config') or {})
    if not options:
        return
    command_line = set()
    if HydraConfig.initialized():
        command_line = {override.split('=', 1)[0].lstrip('+~') for override in HydraConfig.get().overrides.task}
    with open_dict(args):
        for key, value in options.items():
            if f'sampler.pcdaps_config.{key}' not in command_line:
                args.sampler.pcdaps_config[key] = value


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)


def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, record, batch_size, gt, args, root, run_id):
    """
        posterior sampling in batch; PC-DAPS returns a bag of particles [B, K, C, H, W] per measurement
    """
    samples = []
    trajs = []
    prune_log = []
    bag_size = getattr(sampler, 'bag_size', None)
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        e = min(len(x_start), s+batch_size)
        cur_x_start = x_start[s:e]
        cur_y = y[s:e]
        cur_gt = gt[s: e]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, verbose=verbose, record=record, gt=cur_gt)

        samples.append(cur_samples)
        if record:
            cur_trajs = sampler.trajectory.compile()
            trajs.append(cur_trajs)

        # PC-DAPS pruning events with their global image index
        for entry in getattr(sampler, 'prune_log', []):
            prune_log.append({'image': entry['bag'] + s, 'run': run_id, **entry})

        # (image index, particle index, file name) of every sample of the batch
        if bag_size:
            instances = [(idx, k, '{:05d}_run{:04d}_p{:02d}'.format(idx+s, run_id, k)) for idx in range(e-s) for k in range(bag_size)]
        else:
            instances = [(idx, None, '{:05d}_run{:04d}'.format(idx+s, run_id)) for idx in range(e-s)]

        # log individual sample instances
        if args.save_samples:
            image_dir = safe_dir(root / 'samples')
            for idx, k, file_name in instances:
                sample = cur_samples[idx] if k is None else cur_samples[idx, k]
                tensor_to_pils(sample[None])[0].save(str(image_dir / (file_name + '.png')))

        # log sampling trajectory and mp4 video
        if args.save_traj:
            traj_dir = safe_dir(root / 'trajectory')
            # save mp4 video for trajectories
            x0hat_traj = cur_trajs.tensor_data['x0hat']
            x0y_traj = cur_trajs.tensor_data['x0y']
            xt_traj = cur_trajs.tensor_data['xt']
            cur_resized_y = resize(cur_y, cur_samples, args.task[args.task_group].operator.name)
            slices = np.linspace(0, len(x0hat_traj)-1, 10).astype(int)
            slices = np.unique(slices)
            for idx, k, file_name in instances:
                index = (slice(None), idx) if k is None else (slice(None), idx, k)
                sample = cur_samples[idx] if k is None else cur_samples[idx, k]
                if args.save_traj_video:
                    video_path = str(traj_dir / (file_name + '.mp4'))
                    save_mp4_video(sample, cur_resized_y[idx], x0hat_traj[index], x0y_traj[index], xt_traj[index], video_path)
                # save long grid images
                selected_traj_grid = torch.cat([x0y_traj[index][slices], x0hat_traj[index][slices], xt_traj[index][slices]], dim=0)
                traj_grid_path = str(traj_dir / (file_name + '.png'))
                save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))

    if record:
        trajs = Trajectory.merge(trajs)
    return torch.cat(samples, dim=0), trajs, prune_log


@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')
def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))
    apply_task_pcdaps_config(args)

    setproctitle.setproctitle(args.name)
    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # get data
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # get operator & measurement
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # get sampler
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)
    if args.sampler.get('name') == 'pcdaps' and task_group.operator.name == 'phase_retrieval' \
            and 'imagenet' in str(args.data.get('root', '')):
        sampler.orient = True

    # get model
    model = get_model(**args.model)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # log hyperparameters and configurations
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # logging to wandb
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    # +++++++++++++++++++++++++++++++++++
    # main sampling process
    bag_size = getattr(sampler, 'bag_size', None)
    full_samples = []
    full_trajs = []
    prune_log = []
    for r in range(args.num_runs):
        print(f'Run: {r}')
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs, run_prune_log = sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose=True, record=args.save_traj,
                                                        batch_size=args.batch_size, gt=images, args=args, root=root, run_id=r)
        # every run gives one sample per image, or a bag of K particles per image for PC-DAPS
        full_samples.append(samples.transpose(0, 1) if bag_size else samples[None])
        full_trajs.append(trajs)
        prune_log.extend(run_prune_log)
    full_samples = torch.cat(full_samples, dim=0)  # [num_samples, B, C, H, W]
    num_samples = full_samples.shape[0]

    # evaluate and log metrics
    results = evaluator.report(images, y, full_samples)
    if args.wandb:
        evaluator.log_wandb(results, args.batch_size)
    markdown_text = evaluator.display(results)
    if num_samples > 1:
        markdown_text = f'\nmetrics of every sample, averaged over the {num_samples} samples of each image:' + markdown_text

    # phase retrieval: faithful / upside-down or color-inverted / failed samples
    if task_group.operator.name == 'phase_retrieval':
        outcomes = pr_outcomes.summarize(full_samples, images)
        json.dump(outcomes, open(str(root / 'pr_outcomes.json'), 'w'), indent=4)
        markdown_text += '\n\n' + pr_outcomes.format_summary(outcomes)
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(markdown_text)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)
    print(markdown_text)

    # PC-DAPS: pruning events, or gamma calibration from within-cluster score gaps (App. A.4)
    if bag_size:
        json.dump(prune_log, open(str(root / 'prune_log.json'), 'w'), indent=4)
        calibration = {}
        for stage in range(len(sampler.stages)):
            entries = [entry for entry in prune_log if entry['stage'] == stage]
            if sampler.calibrate:
                gaps = [gap for entry in entries for gap in entry['gaps']]
                gamma = float(np.percentile(gaps, 95)) if gaps else None
                calibration[f'stage{stage}'] = {'gamma': gamma, 'num_pairs': len(gaps), 'gaps': gaps}
                print(f'pruning stage {stage}: calibrated gamma (95th percentile of {len(gaps)} within-cluster score gaps): {gamma}')
            else:
                num_pruned = sum(not all(entry['kept_clusters']) for entry in entries)
                print(f'pruning stage {stage}: pruned {num_pruned} of {len(images) * args.num_runs} bags ({len(entries)} checks)')
        if sampler.calibrate:
            json.dump(calibration, open(str(root / 'gamma_calibration.json'), 'w'), indent=4)

    # log grid results
    resized_y = resize(y, images, args.task[args.task_group].operator.name)
    stack = torch.cat([images, resized_y, full_samples.flatten(0, 1)])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)
    
    # save raw trajectory data
    if args.save_traj_raw_data:
        traj_dir = safe_dir(root / 'trajectory')
        for run, sde_traj in enumerate(full_trajs):
            # might be SUPER LARGE
            print(f'saving trajectory run {run}...')
            traj_raw_data = safe_dir(traj_dir / 'raw')
            torch.save(sde_traj, str(traj_raw_data / 'trajectory_run{:04d}.pth'.format(run)))
    
    print(f'finish {args.name}!')


if __name__ == '__main__':
    main()