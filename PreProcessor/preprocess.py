from os.path import dirname
from scipy import ndimage
from Common.func import list_of_path_files
import os
from concurrent.futures.thread import ThreadPoolExecutor
import pandas as pd
import nibabel as nib
import numpy as np


def zscore_norm_cli(path):
    """Function for cli: normalize one or several images"""

    def process_one(path1, path_img_out1):
        nib_data1 = nib.load(path1)
        data1 = nib_data1.get_fdata()
        new_data1 = zscore_normalize(data1)
        new_nib1 = nib.Nifti1Image(new_data1.astype(np.float64), nib_data1.affine)
        nib.save(new_nib1, path_img_out1)

    if not os.path.isfile(path): # if is a direcotry

        # create direcotry where save the results
        if path.endswith('/'): path_out = dirname(dirname(path)) + '/normalized'
        else: path_out = dirname(path) + '/normalized'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        # absolut paths of input and output images
        path_imgs = [os.path.join(path, filename) for filename in os.listdir(path)]
        path_imgs_out = [os.path.join(path_out, filename) for filename in os.listdir(path)]
        tot = len(path_imgs)
        for i, (path_img, path_img_out) in enumerate(zip(path_imgs, path_imgs_out)):
            print(f"Subject {i+1}/{tot}:  {path_img}")
            path_img_out = path_img_out.replace(".nii.gz", "_normZscore.nii.gz")
            process_one(path_img, path_img_out)

    else: # if is an image

        # create direcotry where save the results
        path_out = dirname(dirname(path)) + '/normalized'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        filename = os.path.basename(path)
        process_one(path, os.path.join(path_out, filename))
    return path_out


def zscore_normalize(img_data, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        mask (nibabel.nifti1.Nifti1Image): brain mask for img
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    # img_data = img.get_fdata()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_fdata()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        # mask_data = img_data > img_data.mean()
        mask_data = img_data > 0
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    # normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    # return normalized
    return (img_data - mean) / std


def zscore_norm_all_imgs():
    """ Normalization for images saved in a directory structure like HCP """
    path = '/media/lau/datos_facultad/pfc_final/Datos'
    file = 'T1w_acpc_dc_restore_brain.nii.gz'
    new_file = 'T1w_acpc_dc_restore_brain_normZscores.nii.gz'

    folders = ['Antonia', 'Test', 'Val', 'Train']
    reg = 'T1w'

    list_list_origin_path, list_list_id_pacientes = list_of_path_files(path, folders, reg)
    
    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)
    
        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')
            path_file = os.path.join(origin_path, file)
            path_new_file = os.path.join(origin_path, new_file)

            if not os.path.exists(path_new_file):
                nib_data = nib.load(path_file)
                data = nib_data.get_fdata()

                # normalizar
                new_data = zscore_normalize(data)

                new_nib = nib.Nifti1Image(new_data.astype(np.float64), nib_data.affine)
                nib.save(new_nib, path_new_file)
                print('guardado')
            else:
                print('ya existe')


def delete_nonsulci_segmentations():
    """ Assigns zero label to all non-sulci destrieux segmentations """
    # ------------------------
    # direccion base donde estan los Datos
    path = '/media/lau/datos_facultad/pfc_final/Datos/'
    # carpetas en las cuales estarán los archivos a transformar
    folders = ['Antonia', 'Test', 'Train', 'Val']
    # T1w o MNINonLinear
    reg = 'T1w'
    # destrieux original con sus etiquetas en ambos hemisf
    destr_orig_lbls = 'aparc.a2009s+aseg.nii.gz'
    # destrieux re-etiquetado segun lut
    new_destr_lbls = 'aparc.a2009s+aseg_sulcus_labels.nii.gz'
    # ------------------------
    
    list_list_origin_path, list_list_id_pacientes = list_of_path_files(path, folders, reg)
    
    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)
    
        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')
    
            path_file = os.path.join(origin_path, destr_orig_lbls)
            path_new_file = os.path.join(origin_path, new_destr_lbls)

            if not os.path.exists(path_new_file):
                nib_data = nib.load(path_file)
                data = nib_data.get_fdata()

                # quedarme solo con los surcos y hacer cero el resto de estructuras segmentadas
                new_data = data.copy()
                cond = (data < 11139) | (data > 12175) | ((data > 11175) & (data < 12139))
                new_data[cond] = 0

                new_nib = nib.Nifti1Image(new_data.astype('uint16'), nib_data.affine)
                nib.save(new_nib, path_new_file)
                print('creado')
            else:
                print('ya existe')


def relabel(i, cant_pacientes, id_paciente, origin_path, dest_path, lut_destrieux, colum_label, colum_new_label):
    if not os.path.exists(dest_path):
        nib_atlas = nib.load(origin_path)

        ####### re-etiquetar
        data_atlas = nib_atlas.get_fdata().astype('uint16')

        lbls_data = np.unique(data_atlas)[1:]  # no tomo la etiqueta 0
        lbls_lut = lut_destrieux[colum_label]  # todas las etiquetas que puede haber

        new_data_atlas = np.zeros_like(data_atlas)
        for lbl in lbls_lut:
            new_label = lut_destrieux[lut_destrieux[colum_label] == lbl][colum_new_label].iloc[0].astype('uint16')
            new_data_atlas[data_atlas == lbl] = new_label

            new_nib_atlas = nib.Nifti1Image(new_data_atlas, nib_atlas.affine)
            nib.save(new_nib_atlas, dest_path)
        print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')
        print('creado')
    else:
        print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')
        print('ya existe')


def helper_relabel(args):
    """ Function that allows running in parallel """
    relabel(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])


def relabel_destrieux():
    """ Relabel the destrieux segmentations according to two columns of the lut (original_label -> new_label) """
    # ------------------------
    # direccion base donde estan los Datos
    path = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    # carpetas en las cuales estarán los archivos a transformar
    # folders = ['Antonia', 'Test', 'Val', 'Train']
    folders = ['Antonia']
    # T1w o MNINonLinear
    reg = 'MNINonLinear'
    # nombre del archivo .csv de la lut que se quiere usar para transformar: etiqueta_original --> new_label
    # lut_filename = 'lut_Destrieux_fusionado.csv' # lut_Destrieux.csv
    lut_filename = 'LUT_Destrieux.csv'  # lut_Destrieux.csv
    colum_label = "etiq_original"
    colum_new_label = 'etiq_sim_hemis'  # etiqueta_sim, etiq_sim_hemis
    # destrieux original con sus etiquetas en ambos hemisf
    destr_orig_lbls = 'aparc.a2009s+aseg_sulcus_labels.nii.gz'
    # destrieux re-etiquetado segun lut
    # new_destr_lbls = 'aparc.a2009s+aseg_primary_sulcus_labels_fusionados.nii.gz'
    new_destr_lbls = 'aparc.a2009s+aseg_sulcus_labels_fusionados_2.nii.gz'
    # ------------------------

    lut_destrieux = pd.read_csv(path + lut_filename, delimiter=',')

    list_list_origin_path, _, list_list_id_pacientes = list_of_path_files(path, path, folders, reg)

    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)

        args = []
        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            origin_path1 = os.path.join(origin_path, destr_orig_lbls)
            dest_path1 = os.path.join(origin_path, new_destr_lbls)

            ### Por c/paciente
            args.append((i, cant_pacientes, id_paciente, origin_path1, dest_path1, lut_destrieux, colum_label, colum_new_label))

        with ThreadPoolExecutor(max_workers=4) as thread:
            thread.map(helper_relabel, args)


def relabel_manual():
    """ Change from old_lbl to new_lbl label """
    # ------------------------
    origin_path = '/media/lau/datos_facultad/datos_sinc_local/HCP_1200/'
    filename = 'bin_sul+sul_lbl_fus_geod_prop_remove_1.2_60.nii.gz'
    new_filename = 'bin_sul+sul_lbl_fus_geod_prop_remove_1.2_60.nii.gz'
    folders = ['Antonia', 'Test', 'Train', 'Val']
    reg = 'MNINonLinear'
    old_lbl = 100
    new_lbl = 25
    # ------------------------

    list_list_origin_path, _, list_list_id_pacientes = list_of_path_files(origin_path, origin_path, folders, reg)

    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path,
                                                         list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)

        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            path_file = os.path.join(origin_path, filename)
            path_new_file = os.path.join(origin_path, new_filename)
            nib_data = nib.load(path_file)
            data = nib_data.get_fdata()

            cond = data == old_lbl
            data[cond] = new_lbl

            new_nib = nib.Nifti1Image(data, nib_data.affine)
            nib.save(new_nib, path_new_file)

            print('Guardado:',path_new_file,'   ',old_lbl,'--->',new_lbl)


def main():
    pass

if __name__ == '__main__':
    main()

