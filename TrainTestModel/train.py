import importlib
import math
import random
import numpy as np

from TrainTestModel.Unet import metrics, losses
from TrainTestModel.Unet.dataloaders import get_train_loaders
from TrainTestModel.Unet.config import load_config
from TrainTestModel.Unet.model import get_model
import torch
from TrainTestModel.Unet.trainer import UNet3DTrainer
from TrainTestModel.Unet.utils import get_tensorboard_formatter, get_logger


logger = get_logger('main - train')


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    skip_train_validation = trainer_config.get('skip_train_validation', False)

    validate_after_iters = trainer_config.get('validate_after_iters', 1)
    log_after_iters = trainer_config.get('log_after_iters', 1)
    val_log_after_epoch = trainer_config.get('val_log_after_epoch', True)
    if val_log_after_epoch: # validar y guardar log al final de cada epoca
        validate_after_iters = math.ceil(len( loaders['train'] ))
        log_after_iters = validate_after_iters

    loaders_config = config['loaders']
    whole_image = loaders_config.get('whole_image')
    resize_input_unet = loaders_config.get('resize_input_unet')
    if whole_image:
        logger.info(f'Resize input image to fit in unet: {resize_input_unet}')

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(resume, model, optimizer, lr_scheduler, loss_criterion,eval_criterion,
                                             loaders, whole_image, resize_input_unet,
                                             tensorboard_formatter=tensorboard_formatter,
                                             skip_train_validation=skip_train_validation)
    # elif pre_trained is not None:
    #     # fine-tune a given pre-trained model
    #     return UNet3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
    #                                          eval_criterion, device=config['device'], loaders=loaders,
    #                                          whole_image=whole_image, resize_input_unet=resize_input_unet,
    #                                          max_num_epochs=trainer_config['epochs'],
    #                                          max_num_iterations=trainer_config['iters'],
    #                                          val_log_after_epoch=val_log_after_epoch,
    #                                          validate_after_iters=validate_after_iters,
    #                                          log_after_iters=log_after_iters,
    #                                          eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
    #                                          tensorboard_formatter=tensorboard_formatter,
    #                                          skip_train_validation=skip_train_validation)
    else:
        # start training from scratch
        return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, whole_image, resize_input_unet,
                             trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             val_log_after_epoch=val_log_after_epoch,
                             validate_after_iters=validate_after_iters,
                             log_after_iters=log_after_iters,
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             tensorboard_formatter=tensorboard_formatter,
                             skip_train_validation=skip_train_validation)


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)

    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)

    if class_name == 'ReduceLROnPlateau':
        mode = lr_config.get('mode')
        factor = lr_config.get('factor')
        patience = lr_config.get('patience')
        threshold = lr_config.get('threshold')
        verbose = lr_config.get('verbose', True)
        logger.info(f'Learning rate Scheduler: {class_name},  mode: {mode},  threshold:  {threshold},  factor: {factor},  patience: {patience}')
        return clazz(optimizer, mode=mode, factor=float(factor), patience=int(patience), threshold=float(threshold), verbose=verbose)
    else:
        lr_config['optimizer'] = optimizer
        logger.info(f'Learning rate Scheduler: {class_name}')
        return clazz(**lr_config)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    optimizer_str = optimizer_config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = None
    if optimizer_str == 'SGD':
        momentum = optimizer_config['momentum', 0]
        optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=momentum, weight_decay=float(weight_decay))
        logger.info(f'Optimizer: SGD,  lr: {learning_rate},  momentum: {momentum},  weight decay: {weight_decay}')
    elif optimizer_str == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)) # mirar AdamW
        logger.info(f'Optimizer: AdamW,  lr: {learning_rate},  weight decay: {weight_decay}')
    return optimizer


def cli():
    # parser = argparse.ArgumentParser(prog='train', description='Train a Unet3d by setting its parameters in the train_parameters.yaml file')
    # parser.add_argument('conf_path', type=str, help='Path to the YAML config file')
    # args = parser.parse_args()
    # config = load_config(args.conf_path)
    config = load_config('/media/lau/datos_facultad/PFC/Codigo/pfc_final/TrainTestModel/train_parameters.yaml')
    return config


def train_cli(config):

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        random.seed(manual_seed)  ## seed para RandomFlip
        np.random.seed(manual_seed)

    # load subjects, dataset, and dataloaders
    loaders = get_train_loaders(config, manual_seed)

    # obtener modelo
    model = get_model(config)

    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = torch.nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    model = model.to(device)
    # logger.info(f"Sending the model to '{config['device']}'")

    # obtener omtimizer
    optimizer = _create_optimizer(config, model)

    # # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # obtener loss function
    loss_criterion = losses.get_loss_criterion(config)
    logger.info(f'Loss criterion: {loss_criterion.__class__.__name__}')

    # obtener metrica de evaluacion
    eval_criterion = metrics.get_evaluation_metric(config)
    logger.info(f'Evaluation criterion: {eval_criterion.__class__.__name__}')

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)

    # Start training
    trainer.fit()



if __name__ == "__main__":

    # load parameters
    config = cli()

    train(config)
