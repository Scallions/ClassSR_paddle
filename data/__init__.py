"""create dataset and dataloader"""
import logging
import paddle
from paddle.io import Dataset,DataLoader


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = paddle.distributed.ParallelEnv().world_size
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers*4, drop_last=True)
    else:
        return paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)




def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'LQGT_rcan':
        from data.LQGT_rcan_dataset import LQGTDataset_rcan as D
    elif mode == 'LQ_label':
        from data.LQ_label_dataset import LQ_label_Dataset as D
    # datasets for video restoration
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                            dataset_opt['name']))
    return dataset
