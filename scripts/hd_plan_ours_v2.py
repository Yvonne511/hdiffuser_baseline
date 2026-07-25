import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
# from diffuser.models.hier_diffusion import HierDiffusion
import os
import sys
from diffuser.env_ours.utils import seed
from diffuser.env_ours.venv import SubprocVectorEnv
import gym
import random
from einops import rearrange
import torch
from diffuser.env_ours.utils import aggregate_dct
from tqdm import tqdm

import imageio
import time

def get_flag(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

env_name = get_flag("--dataset", "pusht")

class HLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str =f"config.{env_name}_hl"
    goal_source: str = 'dset'  # "random_state", "dset", "fix_goal"
    n_evals: int = 50
    replan: bool = False
    max_steps: int = 500
    eval_idx: int = 1      # which episode of the n_evals to run; -1 runs all
    seed: int = 99         # planner / diffusion sampling randomness
    task_seed: int = 99    # init & goal state sampling; keep fixed across seeds


hl_args = HLParser().parse_args("plan")


class LLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = f"config.{env_name}_ll"

ll_args = LLParser().parse_args("plan")

goal_source = hl_args.goal_source
n_evals = hl_args.n_evals
frameskip= 1
goal_H = hl_args.horizon*hl_args.jump
## task sampling below must not depend on hl_args.seed, so that episode i is the
## same episode across seeds
seed(hl_args.task_seed)
# import pdb; pdb.set_trace()
hl_args.savepath = hl_args.savepath + "_replan_v2" #TODO: change here!!!
if not os.path.exists(hl_args.savepath):
    os.makedirs(hl_args.savepath)
# ---------------------------------- setup ----------------------------------#

def make_env_and_datasets_ours(dataset_name):
    # load yaml config from conf/env/dataset_name.py
    import yaml
    import hydra
    from omegaconf import OmegaConf
    with open(f"diffuser/conf/env/{dataset_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env_cfg = OmegaConf.create(cfg)

    if env_cfg.name == "wall" or env_cfg.name == "deformable_env" or "point_maze" in env_cfg.name:
        from diffuser.env_ours.serial_vector_env import SerialVectorEnv
        envs = SerialVectorEnv(
            [
                gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )
    else:
        envs = SubprocVectorEnv(
            [
                lambda: gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )

    wrapped_env = gym.make(f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = dataset_name
    dsets, orig_dset = hydra.utils.call(env_cfg.dataset)
    return env, envs, dsets, orig_dset

env, envs, dsets, orig_dset = make_env_and_datasets_ours(hl_args.dataset)
dset = orig_dset['valid']
eval_seed = [hl_args.task_seed * n + 1 for n in range(n_evals)]
eval_idx = list(range(n_evals)) if hl_args.eval_idx < 0 else [hl_args.eval_idx]

def prepare_targets():
    states = []
    actions = []
    observations = []
    
    if goal_source == "random_state" or goal_source == "fix_goal":
        # update env config from val trajs
        observations, states, actions, env_info = (
            sample_traj_segment_from_dset(traj_len=2)
        )
        envs.update_env(env_info)

        # sample random states
        fix_goal = goal_source == "fix_goal"
        rand_init_state, rand_goal_state = envs.sample_random_init_goal_states(
            eval_seed, fix_goal=fix_goal
        )
        if hl_args.dataset == "deformable_env": # take rand init state from dset for deformable envs
            rand_init_state = np.array([x[0] for x in states])

        obs_0, state_0 = envs.prepare(eval_seed, rand_init_state)
        obs_g, state_g = envs.prepare(eval_seed, rand_goal_state)

        # add dim for t
        for k in obs_0.keys():
            obs_0[k] = np.expand_dims(obs_0[k], axis=1)
            obs_g[k] = np.expand_dims(obs_g[k], axis=1)

        obs_0 = obs_0
        obs_g = obs_g
        state_0 = rand_init_state  # (b, d)
        state_g = rand_goal_state
        gt_actions = None
        return obs_0, obs_g, state_0, state_g, gt_actions
    else:
        # update env config from val trajs
        observations, states, actions, env_info = (
            sample_traj_segment_from_dset(traj_len=frameskip * goal_H + 1)
        )
        envs.update_env(env_info)

        # get states from val trajs
        init_state = [x[0] for x in states]
        init_state = np.array(init_state)
        actions = torch.stack(actions)
        if goal_source == "random_action":
            actions = torch.randn_like(actions)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=frameskip)
        # exec_actions = self.data_preprocessor.denormalize_actions(actions)
        exec_actions = actions # actions not normalized in dataloader
        # replay actions in env to get gt obses
        rollout_obses, rollout_states, infos = envs.rollout(
            eval_seed, init_state, exec_actions.numpy()
        )
        obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1)
            for key, arr in rollout_obses.items()
        }
        obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1)
            for key, arr in rollout_obses.items()
        }
        state_0 = init_state  # (b, d)
        state_g = rollout_states[:, -1]  # (b, d)
        gt_actions = wm_actions
        return obs_0, obs_g, state_0, state_g, gt_actions

def sample_traj_segment_from_dset(traj_len):
    states = []
    actions = []
    observations = []
    env_info = []

    # Check if any trajectory is long enough
    valid_traj = [
        i
        for i in range(len(dset))
        if dset.get_seq_length(i) >= traj_len
    ]
    if len(valid_traj) == 0:
        raise ValueError("No trajectory in the dataset is long enough.")

    # sample init_states from dset
    for i in range(n_evals):
        max_offset = -1
        while max_offset < 0:  # filter out traj that are not long enough
            traj_id = random.randint(0, len(dset) - 1)
            obs, act, state, e_info = dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len
        state = state.numpy()
        offset = random.randint(0, max_offset)
        print(f"traj {traj_id}  offset {offset} ")
        obs = {
            key: arr[offset : offset + traj_len]
            for key, arr in obs.items()
        }
        state = state[offset : offset + traj_len]
        act = act[offset : offset + traj_len - 1]
        actions.append(act)
        states.append(state)
        observations.append(obs)
        env_info.append(e_info)
    return observations, states, actions, env_info

obs_0, obs_g, state_0, state_g, gt_actions = prepare_targets()

def combine_and_replace(plan_path, rollout_path, out_path):
    plan_img = imageio.imread(plan_path)
    rollout_img = imageio.imread(rollout_path)
    if plan_img.shape[0] != rollout_img.shape[0]:
        h = max(plan_img.shape[0], rollout_img.shape[0])
        def pad_h(img, h):
            pad = np.zeros((h - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            return np.concatenate([img, pad], axis=0)
        plan_img = pad_h(plan_img, h)
        rollout_img = pad_h(rollout_img, h)
    combined = np.concatenate([plan_img, rollout_img], axis=1)
    imageio.imsave(out_path, combined)
    os.remove(plan_path)
    os.remove(rollout_path)

# ---------------------------------- loading ----------------------------------#


n_samples = 500

loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)


hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
hl_policy = Policy(hl_diffusion, dataset.normalizer)
renderer = hl_diffusion_experiment.renderer

ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)

## targets are fixed by now, so re-seeding here only varies diffusion sampling
seed(hl_args.seed)

# ---------------------------------- main loop ----------------------------------#

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

final_success_rate = []
optimal_success_rate = []
final_state_dist = []
optimal_state_dist = []
final_coverage = []
optimal_coverage = []

eval_times = {}

for i in range(n_evals):
    if i not in eval_idx:
        continue

    eval_start = time.time()
    env.prepare(eval_seed[i], state_0[i])
    env.set_task_goal(state_g[i])

    # ---------------------------------- plan once ----------------------------------#
    hl_cond = {
        0: obs_0['visual'][i, 0],
        hl_diffusion.horizon - 1: obs_g['visual'][i, 0],
    }
    action, samples = hl_policy(hl_cond, batch_size=1)
    hl_plan = samples.observations
    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        ll_args.horizon - 1: ll_cond_[:, -1],
    }
    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_actions = ll_samples.actions.reshape(B, (M-1), hl_args.jump + 1, -1)
    ll_actions_seq = ll_actions[:, :, :hl_args.jump]
    ll_action_seq = ll_actions_seq.reshape(B, (M-1) * hl_args.jump, -1)[0]

    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )
    ll_sequence = ll_samples[0]

    fullpath = join(hl_args.savepath, f'{i}.png')
    renderer.composite(fullpath, ll_samples, ncol=1)

    observation = obs_0['visual'][i, 0]

    all_planned_actions = [to_numpy(ll_action_seq)]
    trajectory_states = []

    obses = []
    rewards = []
    dones = []
    infos = []
    success = []
    state_dist = []
    coverage = []
    visuals = []
    cur_goal = obs_g['rgb_array'][i, 0]

    for t in tqdm(range(hl_args.max_steps), desc="Env Steps"):
        if hl_args.replan:
            if t == 0: action = ll_action_seq[0]
            else:
                hl_cond[0] = observation
                _, hl_traj = hl_policy(hl_cond, 1)
                hl_state = hl_traj.observations

                B, M = hl_state.shape[:2]
                ll_cond_ = np.stack([hl_state[:, :-1], hl_state[:, 1:]], axis=2)
                ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)[0]

                ll_cond = {
                    0: ll_cond_[0],
                    hl_args.jump: ll_cond_[-1],
                }
                action, trajectories = ll_policy(ll_cond, 1)
                all_planned_actions.append(to_numpy(trajectories.actions))
        else:
            if t >= len(ll_action_seq): break
            action = ll_action_seq[t]

        o, r, d, info = env.step(action)
        observation = o['visual']
        obses.append(o)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
        visual = np.concatenate([o['rgb_array'], cur_goal], axis=1)
        visuals.append(visual)
        if isinstance(o['visual'], torch.Tensor):
            o['visual'] = o['visual'].numpy()
        eval_result = env.eval_state(state_g[i], o['visual'])
        success.append(eval_result['success'])
        if hl_args.dataset == 'pusht': coverage.append(info['final_coverage'])
        state_dist.append(eval_result['state_dist'])
        cur_state = info['state'] if 'state' in info else observation
        trajectory_states.append(to_numpy(cur_state))
        # if eval_result['success']:
        #     print(f"Trial {i} succeeds, terminating at time {t}")
        #     break
    obses = aggregate_dct(obses)
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    infos = aggregate_dct(infos)
    final_success_rate.append(success[-1])
    optimal_success_rate.append(np.any(success))
    final_state_dist.append(state_dist[-1])
    optimal_state_dist.append(np.min(state_dist))

    frames = np.stack(visuals).astype(np.uint8)
    print("### num frames", frames.shape)
    imageio.mimwrite(join(hl_args.savepath, f'{i}_rollout_success_{np.any(success)}.mp4'), frames, fps=30)

    if hl_args.dataset == 'pusht':
        final_coverage.append(coverage[-1])
        optimal_coverage.append(np.max(coverage))

    if isinstance(obses['visual'], torch.Tensor):
        rollout = obses['visual'].unsqueeze(0).numpy()
    else:
        rollout = torch.from_numpy(obses['visual']).unsqueeze(0).numpy()
    rollout_path = join(hl_args.savepath, f'{i}_rollout.png')
    renderer.composite(rollout_path, rollout, ncol=1)
    combine_and_replace(
        join(hl_args.savepath, f'{i}.png'),
        rollout_path,
        join(hl_args.savepath, f'{i}_combined.png'),
    )

    eval_times[i] = time.time() - eval_start
    np.save(join(hl_args.savepath, f'eval_{i}_planned_actions.npy'), np.array(all_planned_actions, dtype=object))
    np.save(join(hl_args.savepath, f'eval_{i}_trajectory_states.npy'), np.stack(trajectory_states))

results = {
    "final_success_rate": np.mean(final_success_rate),
    "optimal_success_rate": np.mean(optimal_success_rate),
    "final_state_dist": np.mean(final_state_dist),
    "optimal_state_dist": np.mean(optimal_state_dist),
    "final_coverage": np.mean(final_coverage) if len(final_coverage)!=0 else 0, 
    "optimal_coverage": np.mean(optimal_coverage) if len(optimal_coverage)!=0 else 0,
}

# save logs
with open(join(hl_args.savepath, 'eval_logs.json'), 'w') as f:
    json.dump(results, f, indent=4, default=float)

with open(join(hl_args.savepath, 'eval_times.json'), 'w') as f:
    json.dump({str(k): v for k, v in eval_times.items()}, f, indent=4)
