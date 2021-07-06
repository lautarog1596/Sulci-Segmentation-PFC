import os

import torch
import torch.nn as nn
import torchio
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from TrainTestModel.Unet.utils import get_logger
from TrainTestModel.Unet import utils
from torch.nn import functional as F

logger = get_logger(__name__)


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        whole_image (bool): train with whole image or patches
        resize_input_unet (list): new size of images
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        val_log_after_epoch (bool): If True, it is validated and logs are saved at the end of each epoch and the following 2 parameters have no effect
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, whole_image, resize_input_unet, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5, val_log_after_epoch=True,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, skip_train_validation=False):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.whole_image = whole_image
        self.resize_input_unet = resize_input_unet
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.val_log_after_epoch = val_log_after_epoch
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.skip_train_validation = skip_train_validation
        
        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs_tb'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.output_channels = model.classifier.conv_layer.out_channels # da la cantidad de clases o etiquetas
        self.actual_lr = self.optimizer.param_groups[0]['lr']
        self.scaler = torch.cuda.amp.GradScaler()  # mixed precision
        
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')
        

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        whole_image, resize_input_unet, tensorboard_formatter=None, skip_train_validation=False):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, whole_image, resize_input_unet, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   val_log_after_epoch=state['val_log_after_epoch'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation)

    # @classmethod
    # def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
    #                     device, loaders, whole_image, resize_input_unet,
    #                     max_num_epochs=100, max_num_iterations=1e5,
    #                     val_log_after_epoch=True,
    #                     validate_after_iters=100, log_after_iters=100,
    #                     validate_iters=None, num_iterations=1, num_epoch=0,
    #                     eval_score_higher_is_better=True, best_eval_score=None,
    #                     tensorboard_formatter=None, skip_train_validation=False):
    #     logger.info(f"Logging pre-trained model from '{pre_trained}'...")
    #     utils.load_checkpoint(pre_trained, model, None)
    #     checkpoint_dir = os.path.split(pre_trained)[0]
    #     return cls(model, optimizer, lr_scheduler,
    #                loss_criterion, eval_criterion,
    #                device, loaders, whole_image, resize_input_unet, checkpoint_dir,
    #                eval_score_higher_is_better=eval_score_higher_is_better,
    #                best_eval_score=best_eval_score,
    #                num_iterations=num_iterations,
    #                num_epoch=num_epoch,
    #                max_num_epochs=max_num_epochs,
    #                max_num_iterations=max_num_iterations,
    #                val_log_after_epoch=val_log_after_epoch,
    #                validate_after_iters=validate_after_iters,
    #                log_after_iters=log_after_iters,
    #                validate_iters=validate_iters,
    #                tensorboard_formatter=tensorboard_formatter,
    #                skip_train_validation=skip_train_validation)


    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])
            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return
            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")


    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):  # itera cant_sujetos // batch_size * samples_per_volume
            if i % 100 == 0:  # loggear cada 100
                logger.info(f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            inputs, targets, weights = self._split_training_batch(t)
            # utils.plot_slice(inputs, 0)
            outputs_logits, loss = self._forward_pass(inputs, targets, weights)
            train_losses.update(loss.item(), self._batch_size(inputs))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()  # mixed precision
            self.scaler.step(self.optimizer)  # mixed precision
            self.scaler.update()  # mixed precision

            if self.num_iterations % self.validate_after_iters == 0:  # evaluar modelo despues de x iteraciones (en cada epoca)
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val']) # itera cant_sujetos // batch_size * samples_per_volume
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # si se reduce el lr, actualizar el actual y guardar dicho cambio en logger
                if self.actual_lr != self.optimizer.param_groups[0]['lr']:
                    logger.info(f"Decreasing learning rate: {self.actual_lr} ---> {self.optimizer.param_groups[0]['lr']}")
                    self.actual_lr = self.optimizer.param_groups[0]['lr']
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:  # guardar en log_tb luego de x iteraciones
                # # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # # the evaluation metric as well as images in tensorboard will be incorrectly computed
                # if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #     output = self.model.final_activation(output)
                outputs = F.softmax(outputs_logits, dim=1)   # tiene que estar activado xq siempre mi modelo devuelve logits, no probabilidades
                # para problemas de regrecion se usa logits.. para segmentacion probabilidades, sigmoid para binario y
                # softmax para multiclase, aunque esta última generaliza la primera y se suele usar siempre

                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(outputs.float(), targets.float())
                    train_eval_scores.update(eval_score.item(), self._batch_size(inputs))

                # log stats, params and images
                logger.info(f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                self._log_images(inputs, targets, outputs, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False


    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        if self.actual_lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False


    def validate(self, val_loader):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                if i % 100 == 0:  # loggear cada 5
                    logger.info(f'Validation iteration {i}')

                inputs, targets, weights = self._split_training_batch(t)

                output_logits, loss = self._forward_pass(inputs, targets, weights)
                val_losses.update(loss.item(), self._batch_size(inputs))

                # # if model contains final_activation layer for normalizing logits apply it, otherwise
                # # the evaluation metric will be incorrectly computed
                # if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #     output = self.model.final_activation(output)
                output = F.softmax(output_logits, dim=1)  # tiene que estar activado xq siempre mi modelo devuelve logits, no probabilidades

                if i % 100 == 0:  # guardar imágenes cada 100 iteraciones
                    self._log_images(inputs, targets, output, 'val_')

                eval_score = self.eval_criterion(output.float(), targets.float())
                val_scores.update(eval_score.item(), self._batch_size(inputs))

                if self.validate_iters is not None and self.validate_iters <= i:
                    break # stop validation

            self._log_stats('val', val_losses.avg, val_scores.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg


    def _split_training_batch(self, t):
        weights = None

        inputs = t['image'][torchio.DATA].to(self.device, non_blocking=True).float()
        targets = t['label'][torchio.DATA].to(self.device, non_blocking=True)

        if self.whole_image:
            # si se entrena con toda la imagen, achicarla para no tener problemas con la Unet
            inputs = torch.nn.functional.interpolate(inputs, size=self.resize_input_unet, mode='trilinear', align_corners=False)
            targets = torch.nn.functional.interpolate(targets, size=self.resize_input_unet, mode='nearest')

        # weights = weights.data + 1
        # Debería ir en dataloader, pero de esa manera carga todas las imágenes en la RAM y se necesitarían como unos 60 gb
        # Sumo 1 para que el background quede en 1 y el foreground en 2
        # de esta manera, el mapa de probabilidades queda con el doble de posibilidad de que salga un patch
        # con centro en la etiqueta (este caso surco)

        targets = torch.squeeze(targets, 1)  # [B, 1, Spacial_dimention] --> [B, Spacial_dimention]
        targets = utils.expand_as_one_hot(targets.long(), self.output_channels).float()

        return inputs, targets, weights


    def _forward_pass(self, inputs, targets, weights=None):
        with torch.cuda.amp.autocast():  # fp32 -> fp16
            # forward pass
            outputs_logits = self.model(inputs)
            # compute the loss
            if weights is None:
                loss = self.loss_criterion(outputs_logits, targets)
            else:
                loss = self.loss_criterion(outputs_logits, targets, weights)
        return outputs_logits, loss


    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score
        return is_best


    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations + 1,
            'model_state_dict': state_dict,
            # 'lr_scheduler' : self.scheduler,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'val_log_after_epoch' : self.val_log_after_epoch,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir, logger=logger)


    def _get_num_after_iterations_to_log(self):
        if self.val_log_after_epoch: return self.num_epoch
        else: return self.num_iterations


    def _log_lr(self):
        num = self._get_num_after_iterations_to_log()
        self.writer.add_scalar('learning_rate', self.actual_lr, num)


    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }
        num = self._get_num_after_iterations_to_log()
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, num)


    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        num = self._get_num_after_iterations_to_log()
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), num)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), num)


    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()
        num = self._get_num_after_iterations_to_log()
        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, num, dataformats='CHW')


    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
