import json
import os
import time
from collections import defaultdict
import csv

import torch
from termcolor import colored
import re

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), 
            ('step', 'S', 'int'),
            ('duration', 'D', 'time'), 
            ('episode_reward', 'R', 'float'),
            ('actor_loss', 'AL', 'float'), 
            ('critic_loss', 'CL', 'float'),
            ('Q_distance', 'Dis', 'float'),
            ('Q_list', 'l', 'int'),
        ],
        'eval_train': [
            ('step', 'S', 'int'), 
            ('episode', 'E', 'int'),
            ('episode_reward_train', 'ER', 'float'),
        ],
        'eval_colorhard': [
            ('step', 'S', 'int'),
            ('episode', 'E', 'int'),
            ('episode_reward_colorhard', 'ER', 'float'),
        ],
        'eval_videoeasy': [
            ('step', 'S', 'int'),
            ('episode', 'E', 'int'),
            ('episode_reward_videoeasy', 'ER', 'float'),
        ],
        'eval_videohard': [
            ('step', 'S', 'int'),
            ('episode', 'E', 'int'),
            ('episode_reward_videohard', 'ER', 'float'),
        ]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)  # AverageMeter.update

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            elif key.startswith('eval_train'):
                key = key[len('eval_train') + 1:]
            elif key.startswith('eval_colorhard'):
                key = key[len('eval_colorhard') + 1:]
            elif key.startswith('eval_videoeasy'):
                key = key[len('eval_videoeasy') + 1:]
            elif key.startswith('eval_videohard'):
                key = key[len('eval_videohard') + 1:]
            else:
                raise ValueError('invalid key: %s' % key)
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')


    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        if prefix == 'train':
            prefix = colored(prefix, 'yellow')
        elif prefix == 'eval_colorhard':
            prefix = colored(prefix, 'green')
        elif prefix == 'eval_videoeasy':
            prefix = colored(prefix, 'blue')
        elif prefix == 'eval_videohard':
            prefix = colored(prefix, 'red')
        elif prefix == 'eval_train':
            prefix = colored(prefix, 'magenta')

        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, args, config='rl'):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(
            os.path.join(log_dir, f'{args.seed}train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_train_mg = MetersGroup(
            os.path.join(log_dir, f'{args.seed}eval_train.log'),
            formating=FORMAT_CONFIG[config]['eval_train']
        )
        self._eval_colorhard_mg = MetersGroup(
            os.path.join(log_dir, f'{args.seed}eval_colorhard.log'),
            formating=FORMAT_CONFIG[config]['eval_colorhard']
        )
        self._eval_videoeasy_mg = MetersGroup(
            os.path.join(log_dir, f'{args.seed}eval_videoeasy.log'),
            formating=FORMAT_CONFIG[config]['eval_videoeasy']
        )
        self._eval_videohard_mg = MetersGroup(
            os.path.join(log_dir, f'{args.seed}eval_videohard.log'),
            formating=FORMAT_CONFIG[config]['eval_videohard']
        )



    def log(self, key, value, step, n=1):
        if type(value) == torch.Tensor:
            value = value.item()
        match = re.match(r'(train|eval)(_train|_colorhard|_videoeasy|_videohard)?/', key)
        if match:
            mg_type = match.group(1)
            eval_type = match.group(2)
            if mg_type == 'train':
                mg = self._train_mg
            elif mg_type == 'eval':
                if eval_type == '_colorhard':
                    mg = self._eval_colorhard_mg
                elif eval_type == '_videoeasy':
                    mg = self._eval_videoeasy_mg
                elif eval_type == '_videohard':
                    mg = self._eval_videohard_mg
                elif eval_type == '_train':
                    mg = self._eval_train_mg
        else:
            raise ValueError('invalid key: %s' % key)
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_train_mg.dump(step, 'eval_train')
        self._eval_colorhard_mg.dump(step, 'eval_colorhard')
        self._eval_videoeasy_mg.dump(step, 'eval_videoeasy')
        self._eval_videohard_mg.dump(step, 'eval_videohard')
