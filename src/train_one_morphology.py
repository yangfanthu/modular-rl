from __future__ import print_function
import numpy as np
import torch
import os
import utils
import TD3
import json
import time
from tensorboardX import SummaryWriter
from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import checkpoint as cp
from config import *
"""
possible names for walker morphologies:
['walker_3_flipped', 'walker_5_main', 'walker_4_main', 'walker_6_flipped', 'walker_5_flipped', 
'walker_2_main', 'walker_3_main', 'walker_7_flipped', 'walker_6_main', 'walker_7_main', 
'walker_2_flipped', 'walker_4_flipped']
"""
def train(args):

    # Set up directories ===========================================================
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    exp_name = "EXP_%04d" % (args.expID)
    exp_path = os.path.join(DATA_DIR, exp_name)
    rb_path = os.path.join(BUFFER_DIR, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(rb_path, exist_ok=True)
    # save arguments
    with open(os.path.join(exp_path, 'args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    # Retrieve MuJoCo XML files for training ========================================
    agent_name = args.agent_name
    envs_train_names = [agent_name]
    args.graphs = dict()
    # existing envs
    if not args.custom_xml:
        args.graphs[agent_name] = utils.getGraphStructure(os.path.join(XML_DIR, '{}.xml'.format(agent_name)))
    # custom envs

    num_envs_train = len(envs_train_names)
    print("#" * 50 + '\ntraining envs: {}\n'.format(envs_train_names) + "#" * 50)

    # Set up training env and policy ================================================
    args.limb_obs_size, args.max_action = utils.registerEnvs(envs_train_names, args.max_episode_steps, args.custom_xml)
    max_num_limbs = max([len(args.graphs[env_name]) for env_name in envs_train_names])
    # create vectorized training env
    obs_max_len = max([len(args.graphs[env_name]) for env_name in envs_train_names]) * args.limb_obs_size
    envs_train = [utils.makeEnvWrapper(name, obs_max_len, args.seed) for name in envs_train_names]
    # envs_train = SubprocVecEnv(envs_train)  # vectorized env
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # determine the maximum number of children in all the training envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(envs_train_names, args.graphs)
    # setup agent policy
    policy = TD3.LifeLongTD3(args)

    # Create new training instance or load previous checkpoint ========================
    if cp.has_checkpoint(exp_path, rb_path):
        print("*** loading checkpoint from {} ***".format(exp_path))
        total_timesteps, episode_num, replay_buffer, num_samples, loaded_path = cp.load_checkpoint(exp_path, rb_path, policy, args)
        print("*** checkpoint loaded from {} ***".format(loaded_path))
    else:
        print("*** training from scratch ***")
        # init training vars
        total_timesteps = 0
        episode_num = 0
        num_samples = 0
        # different replay buffer for each env; avoid using too much memory if there are too many envs

    # Initialize training variables ================================================
    writer = SummaryWriter("%s/%s/" % (DATA_DIR, exp_name))
    s = time.time()
    # TODO: may have to change the following codes into the loop
    timesteps_since_saving = 0
    this_training_timesteps = 0
    episode_timesteps = 0
    episode_reward = 0
    episode_reward_buffer = 0
    done = True

    # Start training ===========================================================
    for env_handle, env_name in zip(envs_train, envs_train_names):
        env = env_handle()
        obs = env.reset()
        replay_buffer = utils.ReplayBuffer(max_size=args.rb_max)
        policy.change_morphology(args.graphs[env_name])
        policy.graph = args.graphs[env_name]
        task_timesteps=0
        done = False
        episode_timesteps = 0
        episode_reward = 0
        episode_reward_buffer = 0
        while task_timesteps < args.max_timesteps:
            # train and log after one episode for each env
            if done:
                # log updates and train policy
                if this_training_timesteps != 0:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size,
                                args.discount, args.tau, args.policy_noise, args.noise_clip,
                                args.policy_freq, graphs=args.graphs, env_name=env_name)
                    # add to tensorboard display
                    
                    writer.add_scalar('{}_episode_reward'.format(env_name), episode_reward, task_timesteps)
                    writer.add_scalar('{}_episode_len'.format(env_name), episode_timesteps, task_timesteps)
                    # print to console
                    print("-" * 50 + "\nExpID: {}, FPS: {:.2f}, TotalT: {}, EpisodeNum: {}, SampleNum: {}, ReplayBSize: {}".format(
                            args.expID, this_training_timesteps / (time.time() - s),
                            total_timesteps, episode_num, num_samples,
                            len(replay_buffer.storage)))
                    print("{} === EpisodeT: {}, Reward: {:.2f}".format(env_name,
                                                                    episode_timesteps,
                                                                    episode_reward))
                    this_training_timesteps = 0
                    s = time.time()

                # save model and replay buffers
                if timesteps_since_saving >= args.save_freq:
                    print("!!!!!")
                    timesteps_since_saving = 0
                    model_saved_path = cp.save_model(exp_path, policy, total_timesteps,
                                                    episode_num, num_samples, {env_name: replay_buffer},
                                                    envs_train_names, args)
                    print("*** model saved to {} ***".format(model_saved_path))
                    rb_saved_path = cp.save_replay_buffer(rb_path, {env_name: replay_buffer})
                    print("*** replay buffers saved to {} ***".format(rb_saved_path))

                # reset training variables
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                # create reward buffer to store reward for one sub-env when it is not done
                episode_reward_buffer = 0

            # start sampling ===========================================================
            # sample action randomly for sometime and then according to the policy
            if task_timesteps < args.start_timesteps:
                action = np.random.uniform(low=env.action_space.low[0],
                                                high=env.action_space.high[0],
                                                size=max_num_limbs)
            else:
                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(obs[:args.limb_obs_size * len(args.graphs[env_name])])
                policy_action = policy.select_action(obs)
                if args.expl_noise != 0:
                    policy_action = (policy_action + np.random.normal(0, args.expl_noise,
                        size=policy_action.size)).clip(env.action_space.low[0],
                        env.action_space.high[0])
                # add 0-padding to ensure that size is the same for all envs
                action = np.append(policy_action, np.array([0 for i in range(max_num_limbs - policy_action.size)]))

            # perform action in the environment
            new_obs, reward, done, _ = env.step(action)

            # record if each env has ever been 'done'

            # add the instant reward to the cumulative buffer
            # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
            episode_reward_buffer += reward
            if done and episode_reward == 0:
                episode_reward = episode_reward_buffer
                episode_reward_buffer = 0
            writer.add_scalar('{}_instant_reward'.format(env_name), reward, task_timesteps)
            done_bool = float(done)
            if episode_timesteps + 1 == args.max_episode_steps:
                done_bool = 0
                done = True
            # remove 0 padding before storing in the replay buffer (trick for vectorized env)
            num_limbs = len(args.graphs[env_name])
            obs = np.array(obs[:args.limb_obs_size * num_limbs])
            new_obs = np.array(new_obs[:args.limb_obs_size * num_limbs])
            action = np.array(action[:num_limbs])
            # insert transition in the replay buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))
            num_samples += 1
            # do not increment episode_timesteps if the sub-env has been 'done'
            if not done:
                episode_timesteps += 1
                total_timesteps += 1
                task_timesteps += 1
                this_training_timesteps += 1
                timesteps_since_saving += 1

            obs = new_obs
        policy.next_task()

    # save checkpoint after training ===========================================================
    model_saved_path = cp.save_model(exp_path, policy, total_timesteps,
                                     episode_num, num_samples, {envs_train_names[-1]: replay_buffer},
                                     envs_train_names, args)
    print("*** training finished and model saved to {} ***".format(model_saved_path))


if __name__ == "__main__":
    args = get_args()
    train(args)
