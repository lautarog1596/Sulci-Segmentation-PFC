from os.path import dirname
from Common.func import list_of_path_files, descartar_comp_chicos
import os
import pandas as pd
import nibabel as nib
import numpy.linalg as npl
from PostProcessor.Postprocessor.report import comp_con_terc



##################################################################################################

def defuse_cli(path):
    path_lut = os.path.dirname(__file__) # get the directory name of the running file.
    path_lut = dirname(dirname(path_lut)) + '/LUT_Destrieux.csv'
    old_new_nombre_lut = ['etiq_sim_hemis', 'etiqueta_sim']

    # obtener lista con las etiquetas viejas del hemisferio derecho, y las nuevas a asignar
    lut_destrieux = pd.read_csv(path_lut, delimiter=',')
    old_rh_lbls, new_rh_lbls = asign_new_lbls(lut_destrieux, old_new_nombre_lut)

    # get a list of all image paths
    path_imgs = []
    path_imgs_out = []
    path_out = ''
    if not os.path.isfile(path):  # if is a direcotry

        # absolut path of output images
        # path_out = dirname(path) + '/predictions_defused'
        if path.endswith('/'): path_out = dirname(dirname(path)) + '/predictions'
        else: path_out = dirname(path) + '/predictions'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        path_imgs = [os.path.join(path, path_img) for path_img in os.listdir(path)]
        path_imgs_out = [os.path.join(path_out, filename) for filename in os.listdir(path)]
    else:  # if is an image

        # absolut path of output images
        # path_out = dirname(dirname(path)) + '/predictions_defused'
        path_out = dirname(dirname(path)) + '/predictions'
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        path_imgs.append(path)
        filename = os.path.basename(path)
        path_imgs_out.append(os.path.join(path_out, filename))

    tot = len(path_imgs)
    for i, (path_img, path_img_out) in enumerate(zip(path_imgs, path_imgs_out)):
        # print(f"Subject {i+1}/{tot}:  {path_img}")
        # path_img_out = path_img_out.replace(".nii.gz", "_defused.nii.gz")
        re_label(path_img, path_img_out, old_rh_lbls, new_rh_lbls, 25, 100)
        separar_terciarios(path_img_out, path_img_out, 100)

    return path_out

##################################################################################################



def asign_new_lbls(lut_destrieux, old_new_nombre_lut):
    old_lbls = lut_destrieux[old_new_nombre_lut[0]]
    new_lbls = lut_destrieux[old_new_nombre_lut[1]]
    for i, (old, new) in enumerate(zip(old_lbls, new_lbls)):
        if old != new:
            old_lbls = old_lbls[i:]
            new_lbls = new_lbls[i:]
            break
    return old_lbls, new_lbls


def re_label(path_file, path_new_file, old_rh_lbls, new_rh_lbls, etiqueta_terciarios, nueva_etiq_terc):

    # if not os.path.exists(path_new_file):
        nib_fus = nib.load(path_file)

        ####### re-etiquetar
        data_fus = nib_fus.get_fdata().astype('uint16')

        # el surco 25 que son los terciarios, los reetiqueto con el valor 100
        cond = data_fus == etiqueta_terciarios
        data_fus[cond] = nueva_etiq_terc

        # lbls_fus = np.unique(data_fus)[1:]  # no tomo la etiqueta 0

        # coordenadas que cortan el centro del cerebro, corte sagital
        # ---------------
        # Affine gives de map from voxel to scanner, and inverse of affine de scanner to voxel
        affine = nib_fus.affine
        affine_inv = npl.inv(affine)
        xyz_orig = [0, 0, 0]
        ijk_orig = nib.affines.apply_affine(affine_inv, xyz_orig).astype('int')
        # ------------------

        tmp_new_data_defus = data_fus[:ijk_orig[0], :, :] # solo hemisferio derecho, donde re-etiquetar
        for old_lbl, new_lbl in zip(old_rh_lbls, new_rh_lbls):
            tmp_bool_hd = data_fus[:ijk_orig[0], :, :] == old_lbl  # posiciones donde data_fus es == old_lbl en el hd
            tmp_new_data_defus[tmp_bool_hd] = new_lbl
        # unir hi con hd
        new_data_defus = data_fus
        new_data_defus[:ijk_orig[0], :, :] = tmp_new_data_defus

        new_nib_defus = nib.Nifti1Image(new_data_defus, affine)
        nib.save(new_nib_defus, path_new_file)
        # print(f're-etiquetado/defusionado')
    # else:
        # print(f'ya existe')


def separar_terciarios(path_file, path_new_file, etiqueta_terciarios, threshold_vol):

    # if not os.path.exists(path_new_file):
        nibf = nib.load(path_file)

        #### 1- Componentes conectadas de los "terciarios" que tienen etiqueta 100
        dataf = comp_con_terc(nibf, etiqueta_terciarios)

        # descartar surcos pequeños
        nibf = nib.nifti1.Nifti1Image(dataf, nibf.affine)
        nibf = descartar_comp_chicos(nibf, threshold_vol=threshold_vol)
        dataf = nibf.get_fdata()

        # corregir etiquetas debido al descarte anterior..
        dataf[dataf > 99] = etiqueta_terciarios  # los terciarios tenían etiquetas del 100 en adelante
        nibf = nib.nifti1.Nifti1Image(dataf, nibf.affine)
        dataf = comp_con_terc(nibf, etiqueta_terciarios)  # volver a etiquetarlos por separado

        new_nib_defus = nib.Nifti1Image(dataf, nibf.affine)
        nib.save(new_nib_defus, path_new_file)
    #     print(f'terciarios separados')
    # else:
    #     print(f'ya existe')


def main():
    """ Recibe las segmentaciones fucionadas y las separa con las etiquetas según sea hemisferio derecho o izquierdo.
        Las etiquetas 2,3,4,5 y 6 tienen que pasar a ser 7,8,9,10 y 11 en el hemisferio derecho.
    """
    # ------------------------
    path = '/media/lau/datos_facultad/pfc_final/Datos/'
    folders = ['Test']
    # segmentacion fusionada
    pred_fus = 'pred_m2_patch_train11_whole.nii.gz'
    new_pred_defus = 'pred_m2_patch_train11_whole_defus.nii.gz'
    # new_pred_defus = 'bin_sul+sul_lbl_fus_geod_prop_remove_1.2_60_defus.nii.gz'
    reg = 'MNINonLinear'
    path_lut = '/media/lau/datos_facultad/pfc_final/Datos/lut_Destrieux.csv' # lut usada para reasignar etiquetas
    old_new_nombre_lut = ['primarios_fus', 'primarios']
    etiqueta_terciarios = 25
    nueva_etiq_terc = 100
    threshold_vol = 60  # umbral en mm cubicos para borrar segmentaciones de sulsicidad menores. 100 sería 1cm cubico
    # ------------------------


    # obtener lista con las etiquetas viejas del hemisferio derecho, y las nuevas a asignar
    lut_destrieux = pd.read_csv(path_lut, delimiter=',')
    old_rh_lbls, new_rh_lbls = asign_new_lbls(lut_destrieux, old_new_nombre_lut)
    
    list_list_origin_path, _, list_list_id_pacientes = list_of_path_files(path, path, folders, reg)

    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)
        
        # list_id_pacientes = list_id_pacientes[:1]
    
        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')
            
            path_file = os.path.join(origin_path, pred_fus)
            path_new_file = os.path.join(origin_path, new_pred_defus)

            re_label(path_file, path_new_file, old_rh_lbls, new_rh_lbls, etiqueta_terciarios, nueva_etiq_terc)
            separar_terciarios(path_new_file, new_pred_defus, nueva_etiq_terc, threshold_vol)
    


if __name__ == '__main__':
    main()
