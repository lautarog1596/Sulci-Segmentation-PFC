import os
import datetime
from os.path import dirname

import cv2
import torch
import torch.nn as nn
import torchio
import nibabel as nib
from matplotlib import pyplot as plt
from multiprocessing import set_start_method

from unet import UNet

set_start_method('spawn', force=True)

from TrainTestModel.Unet import utils
from TrainTestModel.Unet.config import load_config
from TrainTestModel.Unet.model import get_model
from TrainTestModel.Unet.dataloaders import get_test_subj_or_loader
import numpy as np
from torch.nn import functional as F

logger = utils.get_logger('predict')


####################################################################################
def infer_cli(dataset, model):
    """Function for cli: predict one or several images"""
    n_subjects = len(dataset)

    for i, subj in enumerate(dataset):  # se infiere por cada uno de los sujetos, no por batch de sujetos
        path_img = subj['path_img']
        path_out = subj['path_out']

        print(f"Subject {i+1}/{n_subjects}:  {path_img}")

        path_to_pred_file = path_out.replace(".nii.gz", "_prediction.nii.gz")

        if not os.path.isfile(path_to_pred_file):
            pred = predict(True, [256, 304, 256], model, subj, 1, 1, 1, 'cuda')

            # guardar imágenes resultados
            nib_img = nib.load(path_img)
            pred = nib.Nifti1Image(pred, nib_img.affine, nib_img.header)
            nib.save(pred, path_to_pred_file)


def predict_cli(path):
    """Function for cli: predict one or several images"""
    # get a list of all image paths
    path_imgs = []
    path_imgs_out = []
    path_out = ''
    if not os.path.isfile(path):  # if is a direcotry

        # absolut path of output images
        if path.endswith('/'): path_out = dirname(dirname(path)) + '/predictions'
        else: path_out = dirname(path) + '/predictions'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        path_imgs = [os.path.join(path, path_img) for path_img in os.listdir(path)]
        path_imgs_out = [os.path.join(path_out, filename) for filename in os.listdir(path)]
    else:  # if is an image

        # absolut path of output images
        path_out = dirname(dirname(path)) + '/predictions'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        path_imgs.append(path)
        filename = os.path.basename(path)
        path_imgs_out.append(os.path.join(path_out, filename))

    # create dataset TorhIO
    subjects = []
    for path_img, path_img_out in zip(path_imgs, path_imgs_out):
        subject_dict = {'image': torchio.Image(path_img, torchio.INTENSITY),
                        'path_img': path_img,
                        'path_out': path_img_out}
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    dataset = torchio.SubjectsDataset(subjects)

    # create model
    model = UNet(
        in_channels= 1,
        out_classes= 26,
        dimensions= 3,
        num_encoding_blocks= 5,
        out_channels_first_layer= 6,
        normalization= 'batch',
        pooling_type= 'max',
        upsampling_type= 'conv',
        preactivation= False,
        residual= False,
        padding= 1,
        padding_mode= 'replicate',
        activation= 'ReLU',
        dropout= 0,
        monte_carlo_dropout= 0
    )
    model.to('cuda')

    # load model trained
    utils.load_checkpoint(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/TrainTestModel/model_trained_cli/best_checkpoint.pytorch', model)

    # predict
    infer_cli(dataset, model)

    return path_out
####################################################################################



def save_slice(pred, true, subj_id, output_dir, reg):
    output_dir = os.path.join(output_dir, subj_id, reg)

    logger.info(f'Saving central coronal slice to {output_dir}')
    # k = 130 # tomo una slice en la mitad del eje z. (debería quedar un corte axial en el centro del volumen)
    j = 155  # tomo una slice en la mitad del eje y. (debería quedar un corte coronal en el centro del volumen)
    # i = 155 # tomo una slice en la mitad del eje x. (debería quedar un corte sagital en el centro del volumen)
    slice_pred = pred[:, j, :]  # es lo mismo que hacer: one_batch[IMAGE][DATA][...,k]
    slice_img = true[:, j, :]
    slices = np.hstack((slice_img, slice_pred))
    image_path = os.path.join(output_dir, f'target_and_pred_mid_coronal_slice.png')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  # make sure the directory exists

    cv2.imwrite(image_path, np.uint8(slices*255))


def save_nifty(pred, subj_id, output_dir, subj_target_path, path_to_pred_file, reg):
    if not os.path.isdir(os.path.join(output_dir, subj_id)):
        os.mkdir(os.path.join(output_dir, subj_id))  # make sure the directory exists

    output_dir = os.path.join(output_dir, subj_id, reg)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  # make sure the directory exists

    nib_target = nib.load(subj_target_path)
    pred = nib.Nifti1Image(pred, nib_target.affine, nib_target.header)

    logger.info(f'Saving nifty image to {path_to_pred_file}')
    nib.save(pred, path_to_pred_file)


# def plot_boxplot(val_dc, val_hd, val_iou):
def plot_boxplot(val_dc, val_iou, name_nifty_img):
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Evaluation')
    ax1.boxplot(val_dc)
    ax1.set_title('Dice')
    ax1.set_ylim([0, 1])
    ax2.boxplot(val_iou)
    ax2.set_title('IoU')
    now = datetime.datetime.now()
    nombre = 'boxplot_' + name_nifty_img + '_' + now.strftime("%Y%m%d-%H%M%S") + '.png'
    plt.savefig('checkpoints/boxplots/' + nombre)
    plt.show()


# @profile
def predict(whole_image, resize_input_unet, model, subj, batch_size, patch_size, patch_overlap, device):
    model.eval()

    if whole_image:
        input = subj['image'][torchio.DATA].to(device, non_blocking=True).float()
        original_spatial_size = input.shape[-3:]

        # si se entrena con toda la imagen, achicarla para no tener problemas con la Unet
        input = torch.unsqueeze(input, dim=0)
        input = torch.nn.functional.interpolate(input, size=resize_input_unet, mode='trilinear', align_corners=False)

        with torch.no_grad():
            # predecir
            output_logits = model(input)
            output = F.softmax(output_logits, dim=1)  # transformar en probabilidades

            # volver a dimension original
            output = torch.nn.functional.interpolate(output, size=original_spatial_size, mode='trilinear',align_corners=False)

            output = output.argmax(dim=1, keepdim=True)  # binarizar

            output = torch.squeeze(output, dim=0)
            output = torch.squeeze(output, dim=0)
            output = output.to(torch.int8).cpu().numpy()
            torch.cuda.empty_cache() #### Buenisimo para liberar memoria

    else:
        grid_sampler = torchio.inference.GridSampler(subject=subj, patch_size=patch_size, patch_overlap=patch_overlap, )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        with torch.no_grad():
            for patches_batch in patch_loader:
                input_tensor = patches_batch['image'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]
                output_logits = model(input_tensor)
                output = F.softmax(output_logits, dim=1)
                output = output.argmax(dim=1, keepdim=True)
                aggregator.add_batch(output, locations)

        output = aggregator.get_output_tensor()
        # print('encontradas: ', output.unique())
        output = torch.squeeze(output, dim=0).to(torch.int8).numpy()

    return output


def infer(reg, whole_image, test_loader, output_dir, name_nifty_img, model, batch_size, patch_size, patch_overlap,
          device, resize_input_unet, save_pred_as_nifty, save_pred_as_slice):

    n_subjects = len(test_loader)
    # val_dc = []
    # val_hd = []
    # val_iou = []
    # val_dc_classes = []
    # val_iou_classes = []

    for i, subj in enumerate(test_loader):  # se infiere por cada uno de los sujetos, no por batch de sujetos
        subj_id = subj['subj_id']
        subj_path = subj['img_path']
        if isinstance(subj_path, list):
            subj_path = subj['img_path'][0]

        logger.info(f"Processing subjects: {subj_id} , {i}/{n_subjects}")

        path_to_pred_file = os.path.join(output_dir, subj_id, reg, name_nifty_img)

        if not os.path.isfile(path_to_pred_file):
            pred = predict(whole_image, resize_input_unet, model, subj, batch_size, patch_size, patch_overlap, device)
            # guardar imágenes resultados
            if save_pred_as_nifty:
                save_nifty(pred, subj_id, output_dir, subj_path, path_to_pred_file, reg)
        # else:
        #     pred = nib.load(path_to_pred_file)
        #     pred = pred.get_fdata()

        # target = subj['label'][torchio.DATA]
        # target = torch.squeeze(target, dim=0).numpy().astype(np.uint8)  # saco la dimension batch


        # ##### Evaluar Accuracy
        #
        # ## Sobre todas las segmentaciones juntas
        # val_dc.append(metrics.dc(pred, target))
        # # val_hd.append(metrics.hd(pred, true)) # demora bocha de tiempo
        # val_iou.append(metrics.jc(pred, target))
        #
        # ## Sobre cada clase si es segmentacion multiclase
        # list_dice, list_iou, list_lbls = calc_dice_iou_classes(pred, target)
        # val_dc_classes.append(list_dice)
        # val_iou_classes.append(list_iou)

        # if save_pred_as_slice:
        #     save_slice(pred, target, subj_id, output_dir, reg)  # guarda una imagen con la slice de la prediccion y true correspondiente

    # # plot_boxplot(val_dc, val_hd, val_iou)
    # plot_boxplot(val_dc, val_iou, name_nifty_img)
    #
    # avg_dc = sum(val_dc) / n_subjects
    # avg_iou = sum(val_iou) / n_subjects
    #
    # logger.info(f'Total test subjects: {n_subjects},    avg dice: {avg_dc},    avg IoU: {avg_iou}')


def cli():
    # parser = argparse.ArgumentParser(prog='predict',
    #                                  description='Make predictions using a Unet3D model previously trained with the ' \
    #                                             'train.py tool, setting the parameters in the predict_parameters.yaml file')
    # parser.add_argument('conf_path', type=str, help='Path to the YAML config file')
    # args = parser.parse_args()
    # config = load_config(args.conf_path)
    config = load_config('/media/lau/datos_facultad/PFC/Codigo/pfc_final/TrainTestModel/predict_parameters.yaml')
    return config


def main():
      
    # load parameters
    config = cli()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    # Mostrar cantidad de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model's params: {total_params}")
    logger.info(f"Total model's trainable params: {total_trainable_params}")

    loaders_config = config['loaders']

    # trabajar con las imágenes registradas linealmente o MNINonLinear
    reg = loaders_config.get('reg')

    # guardar resultados
    save_pred_as_nifty = loaders_config.get('save_pred_as_nifty', False)
    save_pred_as_slice = loaders_config.get('save_pred_as_slice', False)
    name_nifty_img = loaders_config.get('name_nifty_img')
    # directorio donde guardar resultados
    output_dir = loaders_config.get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # inferir sobre toda la imagen o por parches
    whole_image = loaders_config.get('whole_image')

    batch_size = loaders_config.get('batch_size', 1)
    n_workers = loaders_config.get('n_workers', 1)
    logger.info(f'n_workers: {n_workers}')

    test_loader = get_test_subj_or_loader(loaders_config)

    resize_input_unet = loaders_config.get('resize_input_unet')
    patch_size = loaders_config.get('patch_size')
    patch_overlap = loaders_config.get('patch_overlap', 4)

    # inferir sobre toda la imagen
    if whole_image:
        batch_size = 1
    # inferir por parches
    else:
        logger.info(f'Batch size: {batch_size}')
        logger.info(f'Patch size: {patch_size}')
        logger.info(f'Patch overlap: {patch_overlap}')

    infer(reg, whole_image, test_loader, output_dir, name_nifty_img, model, batch_size, patch_size, patch_overlap,
          device, resize_input_unet, save_pred_as_nifty, save_pred_as_slice)



if __name__ == '__main__':
    main()
