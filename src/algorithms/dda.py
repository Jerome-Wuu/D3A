import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import utils
import algorithms
from copy import deepcopy
import algorithms.modules as m
from collections import deque
from random import choice
from algorithms.sac import SAC
import augmentations
from PIL import Image
from pylab import *
from segnet.nets.segent import SegNet



class DDA(SAC):
    def __init__(self, obs_shape, action_shape, args, image_dir):
        super().__init__(obs_shape, action_shape, args, image_dir)
        self.batch_size = args.batch_size
        self.image_dir = image_dir
        self.num_classes = args.num_classes
        self.random_aug = args.random_aug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model_path = args.pretrained_model_path
        self.model = self.create_model()

    def create_model(self):
        print("Using device: {}".format(self.device))
        model = SegNet(self.num_classes)
        state_dict = torch.load(self.pretrained_model_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
            '''test wo/mask'''
            obs_ = obs / 255.0
            im1, im2, im3 = torch.chunk(obs_, 3, dim=1)
            output1 = self.model(im1)
            output2 = self.model(im2)
            output3 = self.model(im3)

            mask1 = output1.argmax(1).unsqueeze(1).repeat(1, 3, 1, 1)
            mask2 = output2.argmax(1).unsqueeze(1).repeat(1, 3, 1, 1)
            mask3 = output3.argmax(1).unsqueeze(1).repeat(1, 3, 1, 1)

        mask = torch.cat((mask1, mask2, mask3), dim=1).cuda()

        augmentation_funcs = {
            'random_conv': augmentations.random_conv,
            'random_overlay': augmentations.random_overlay,
            'random_cutout': augmentations.random_cutout,
            'random_cutout_color': augmentations.random_cutout_color,
            'random_grayscale': augmentations.random_grayscale,
            'color_jitter': augmentations.random_color_jitter,
            'random_blur': augmentations.random_blur,
            'random_pepper': augmentations.random_pepper,
        }
        if self.random_aug == 'RA':
            selected_augmentation = choice(list(augmentation_funcs.keys()))
        elif self.random_aug == 'overlay':
            selected_augmentation = 'random_overlay'
        elif self.random_aug == 'conv':
            selected_augmentation = 'random_conv'

        random_aug = augmentation_funcs[selected_augmentation](obs_.clone()) * 255.0
        obs_masked = mask * obs + (1 - mask) * random_aug

        current_Q1, current_Q2 = self.critic(obs_masked, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if L is not None:
            L.log('train/critic_loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

        self.update_critic(replay_buffer, obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
