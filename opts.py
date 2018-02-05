import argparse
import datetime
import json
import os


def parse():
    parser = argparse.ArgumentParser(
        description='BC learning for image classification')

    # General settings
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--save', default='results',
                        help='Directory to save the results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='if cpu only, set negative integer')
    # Learning settings
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--plus', action='store_true', help='Use BC+')
    parser.add_argument('--n_epochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-
                        1, help='Initial learning rate')
    parser.add_argument('--schedule', type=float, nargs='*',
                        default=-1, help='When to divide the LR')
    parser.add_argument('--warmup', type=int, default=-1,
                        help='Number of epochs to warm up')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nobias', type=int, default=-1,
                        help='use bias or not. default False')
    parser.add_argument('--dropout_ratio', '-D', type=float, default=.5,
                        help='dropout ratio. default 0.5')
    parser.add_argument('--resume', default=None, help='snapshot')
    parser.add_argument('--debug', action='store_true')
    opt = parser.parse_args()
    if opt.plus and not opt.BC:
        raise Exception('Using only --plus option is invalid.')

    # Dataset details
    if opt.dataset == 'cifar10':
        opt.n_classes = 10
    else:  # cifar100
        opt.n_classes = 100
    opt.nobias = opt.nobias >= 0

    # Default settings
    default_settings = dict()
    default_settings['cifar10'] = {'n_epochs': 250, 'LR': 0.1, 'schedule': [
        0.4, 0.6, 0.8], 'warmup': 0, 'batch_size': 128}
    default_settings['cifar100'] = {'n_epochs': 250, 'LR': 0.1, 'schedule': [
        0.4, 0.6, 0.8], 'warmup': 0, 'batch_size': 128}
    for key in ['n_epochs', 'LR', 'schedule', 'warmup', 'batch_size']:
        if eval('opt.{}'.format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][key])

    _schedule = opt.schedule
    _schedule = [int(opt.n_epochs * i) for i in _schedule]
    _schedule = [i for i in _schedule if i > opt.warmup]
    opt.schedule = _schedule

    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M')
    debug_flag = '_debug' if opt.debug else ''
    learning = display_info(opt)
    opt.save = os.path.join(
        opt.save, timestamp + debug_flag + '_{}'.format(opt.dataset) + '_' + learning)
    if not os.path.isdir(opt.save):
        os.makedirs(opt.save)
        with open(os.path.join(opt.save, 'opts.json'), 'w') as f:
            json.dump(opt.__dict__, f, indent=4)

    return opt


def display_info(opt):
    if opt.BC:
        if opt.plus:
            learning = 'BC+'
        else:
            learning = 'BC'
    else:
        learning = 'standard'
    opt.learning = learning

    print(json.dumps(opt.__dict__, indent=4))
    return learning
