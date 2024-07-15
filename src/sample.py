import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from PIL import Image
import random
from logger import Logger
from video import VideoRecorder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
    )

    # Create working directory
    work_dir = os.path.join(args.log_dir, args.data_domain)
    print('Working directory:', work_dir)
    utils.make_dir(work_dir)
    JPEGImages_dir = utils.make_dir(os.path.join(work_dir, 'JPEGImages'))
    SegmentationClass_dir = utils.make_dir(os.path.join(work_dir, 'SegmentationClass'))
    ImagesSets_dir = utils.make_dir(os.path.join(work_dir, 'ImagesSets'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
        step=args.step
    )
    cropped_obs_shape = (3 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)


    for step in range(0, args.init_steps):
        obs = env.reset()
        frames = np.array(obs).reshape(-1, 3, 84, 84)
        frame = frames[0].transpose(1, 2, 0)
        file_name = "{}/{}.jpg".format(JPEGImages_dir, step)
        Image.fromarray(np.uint8(frame)).save(file_name)

    print('Completed sample for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
