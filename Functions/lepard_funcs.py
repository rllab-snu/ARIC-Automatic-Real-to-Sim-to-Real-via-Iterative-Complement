import os, torch, json, argparse, shutil
from torch import optim
from easydict import EasyDict as edict
import yaml

from References.Lepard.datasets.dataloader import get_dataloader, get_datasets
from References.Lepard.models.pipeline import Pipeline
from References.Lepard.lib.utils import setup_seed
from References.Lepard.lib.tester import get_trainer
from References.Lepard.models.loss import MatchMotionLoss
from References.Lepard.lib.tictok import Timers
from References.Lepard.configs.models import architectures

setup_seed(0)

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)


# if __name__ == '__main__':
def find_best_transformation_lepard(source_points, target_points):
    # load configs
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=str, help= 'Path to the config file.')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    args.config = "References/Lepard/configs/test/3dmatch.yaml"
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['dataset']+config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['dataset']+config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['dataset']+config['folder'], config['exp_dir'])
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')
    
    # backup the
    if config.mode == 'train':
        os.system(f'cp -r models {config.snapshot_dir}')
        os.system(f'cp -r configs {config.snapshot_dir}')
        os.system(f'cp -r cpp_wrappers {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r kernels {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py',config.snapshot_dir)

    
    # model initialization
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.model = Pipeline(config)
    # config.model = KPFCNN(config)

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    

    #create learning rate scheduler
    if  'overfit' in config.exp_dir :
        config.scheduler = optim.lr_scheduler.MultiStepLR(
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting
            gamma=0.1,
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(
            config.optimizer,
            gamma=config.scheduler_gamma,
        )


    config.timers = Timers()

    # create dataset and dataloader
    train_set, val_set, test_set = get_datasets(config, source_points, target_points)
    config.train_loader, neighborhood_limits = get_dataloader(train_set,config,shuffle=True)
    config.val_loader, _ = get_dataloader(val_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    config.test_loader, _ = get_dataloader(test_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    
    # config.desc_loss = MetricLoss(config)
    config.desc_loss = MatchMotionLoss (config['train_loss'])

    trainer = get_trainer(config)
    # if(config.mode=='train'):
    #     trainer.train()
    # else:
    best_transform_mat, best_score = trainer.test()
    return best_transform_mat, best_score
