from os.path import dirname

from nilearn.regions.region_extractor import connected_label_regions
import pandas as pd
from Common.func import create_dir
from Common.func import load_nibs
from Common.func import list_of_path_files, descartar_comp_chicos
import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from Common.func import plot_slices2

plt.rcParams.update({'figure.max_open_warning': 0})


#########################################################################
def report_cli(path, path_brain):

    path_lut = os.path.dirname(__file__)  # get the directory name of the running file.
    path_lut = dirname(dirname(path_lut)) + '/LUT_Destrieux.csv'
    lut_destrieux = pd.read_csv(path_lut, delimiter=',')

    # get a list of all image paths
    path_imgs = []
    path_imgs_out = []
    path_out = ''
    if not os.path.isfile(path):  # if is a direcotry

        # absolut path of output images
        # path_out = dirname(path) + '/predictions_defused'
        path_out = dirname(path) + '/predictions'
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


    # get a list of all image brain paths
    path_imgs_brain = []
    if not os.path.isfile(path_brain):  # if is a direcotry
        path_imgs_brain = [os.path.join(path_brain, path_img) for path_img in os.listdir(path_brain)]
    else:  # if is an image
        path_imgs_brain.append(path_brain)


    tot = len(path_imgs)
    for i, (path_img, path_img_out, path_img_brain) in enumerate(zip(path_imgs, path_imgs_out, path_imgs_brain)):
        print(f"Subject {i + 1}/{tot}:  {path_img}")

        ## cargar las imágenes
        nib_pred, nib_brain = load_nibs(path_img, path_img_brain)
        affine = nib_pred.affine  # ambas en el espacio MNI

        ## crear directorios donde guardar resultados
        path_reporte = path_img_out.replace(".nii.gz", "")
        path_reporte = create_dir(path_out, os.path.basename(path_reporte) )[0]

        data_pred = nib_pred.get_fdata()

        # cantidad de componentes de cada imagen
        labels_pred = np.unique(data_pred).astype('int')
        # quitar la etiqueta cero ya que corresponde al fondo
        labels_pred = np.delete(labels_pred, 0)

        #### 2- Generar segmentacion en primarios, secundarios y terciarios. Generar cortes de cada surco

        cmap = get_cmap(len(labels_pred), 'hsv')

        data_pred_prim = data_pred.copy()

        for lbl in labels_pred:
            # crear imagen nifty con las segmentaciones en primarios secundarios y terciarios
            classification = lut_destrieux[lut_destrieux.etiqueta_sim == lbl].clasificacion
            if len(classification) > 0:  # primario o secundario
                clas = classification.iloc[0]
            else:  # terciario
                clas = 3
            data_pred_prim[data_pred_prim == lbl] = clas

            # crear cortes por el centro de cada surco y guardarlos según su clasificacion y nombre
            name = lut_destrieux[lut_destrieux.etiqueta_sim == lbl].nom_sim
            if len(name) > 0:  # primario o secundario
                nombre = name.iloc[0]
            else:  # terciario
                nombre = 'terciario'
            f_name = str(clas) + '--' + str(nombre) + '--' + str(lbl)
            plot_sulcus(data_pred, lbl, affine, nib_brain, path_reporte, f_name, cmap)

    return path_out

#########################################################################


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_sulcus(data_pred, lbl_pred, affine, nib_brain, path_save, f_name, cmap):
    bool_anot_lbl = np.equal(data_pred, lbl_pred).astype('int')
    data_interest = np.multiply(data_pred, bool_anot_lbl)
    nib_pred_interest = nib.nifti1.Nifti1Image(data_interest, affine)

    ijk_cut_coords = center_of_mass(data_interest)
    xyz_cut_coords = nib.affines.apply_affine(affine, ijk_cut_coords)

    display = plot_slices2(nib_pred_interest, xyz_cut_coords, nib_brain, nib_img_contour=None, cmap=cmap, titulo=f_name)

    path_save_file = os.path.join(path_save, f_name + '.png')
    display.savefig(path_save_file)
    display.close()
    im = cv2.imread(path_save_file)
    im = im[:-80, 80:-80, :]
    cv2.imwrite(path_save_file, im)


def comp_con_terc(nib_pred, etiqueta_terciarios):
    """Recibe un archivo nibabel de la prediccion con todas las segmentaciones, donde
    el mayor numero es 25 y corresponde a los surcos terciarios, entonces los separa
    en componentes conectadas asignando valores sucesivos mayores

    Args:
        nib_pred (nibabel): Imagen formato nibabel
        etiqueta_terciarios (int): Entero correspondiente a los terciarios

    Returns:
        data_pred (numpy): imagen numpy con los terciarios separados
    """
    
    # me quedo solo con los terciarios
    data_pred = nib_pred.get_fdata()
    data_pred_terc = data_pred.copy()
    data_pred_terc[data_pred_terc != etiqueta_terciarios] = 0
    nib_pred_terc = nib.nifti1.Nifti1Image(data_pred_terc, nib_pred.affine)
    
    # le asigno componentes conectadas a partir de la etiqueta 100          
    nib_pred_terc_cc = connected_label_regions(nib_pred_terc, connect_diag=True)
    # cc_fig = plotting.plot_roi(nib_pred_cc, title='CC predict', colorbar=True, cmap='Paired')
    # cc_fig.savefig(_path_out + 'roi_cc_pred')
    data_pred_terc_cc = nib_pred_terc_cc.get_fdata()
    pos_terc = (data_pred_terc_cc!=0)
    data_pred_terc_cc[pos_terc] = data_pred_terc_cc[pos_terc] + 99

    # junto los terciarios mas el resto y guardo la imagen
    data_pred[pos_terc] = data_pred_terc_cc[pos_terc]
    nib_pred_cc = nib.nifti1.Nifti1Image(data_pred, nib_pred.affine)
    # nib.save(nib_pred_cc, 'nib_pred_cc_PRUEBA.nii.gz')
    
    # obtener data en numpy
    data_pred = nib_pred_cc.get_fdata()
            
    return data_pred
    


def main():

    # ------
    # Parámetros
    path_pred = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    folders = ['Test']
    filename_pred = 'pred_M26_whole_defus.nii.gz'
    reg = 'MNINonLinear'
    path_lut = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/lut_Destrieux.csv' # lut usada para clasificar y obtener nombres de los surcos
    # ------

    lut_destrieux = pd.read_csv(path_lut, delimiter=',')
    list_list_path_pred, _, list_list_id_pacientes = list_of_path_files(path_pred, path_pred, folders, reg)
    

    # por cada carpeta
    for list_path_pred, list_id_pacientes, fold in zip(list_list_path_pred, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        
        list_path_pred = list_path_pred[:1] ########### solo 1 VERRRR
        list_id_pacientes = list_id_pacientes[:1]
        cant_pacientes = len(list_path_pred)

        for i, (path_pred, id_paciente) in enumerate(zip(list_path_pred, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            path_pred_file = os.path.join(path_pred, filename_pred)
            path_brain_file = os.path.join(path_pred, 'T1w_restore_brain.nii.gz')
            
            ## cargar las imágenes
            nib_pred, nib_brain = load_nibs(path_pred_file, path_brain_file)
            affine = nib_pred.affine # ambas en el espacio MNI

            ## crear directorios donde guardar resultados
            path_reporte = create_dir(os.path.dirname(os.path.abspath(__file__)), 'reporte')[0]
            path_reporte_fold = create_dir(path_reporte, fold)[0]
            if os.path.exists(os.path.join(path_reporte_fold, id_paciente)):
                print('Ya existe el directorio!!!!!!!')
                exit()
            path_reporte_fold_id = create_dir(path_reporte_fold, id_paciente)[0]
            path_surcos = create_dir(path_reporte_fold_id, 'surcos_cortes')[0]
            
            data_pred = nib_pred.get_fdata()

            # cantidad de componentes de cada imagen
            labels_pred = np.unique(data_pred).astype('int')
            # quitar la etiqueta cero ya que corresponde al fondo
            labels_pred = np.delete(labels_pred, 0)
            
            #### 2- Generar segmentacion en primarios, secundarios y terciarios. Generar cortes de cada surco

            cmap = get_cmap(len(labels_pred), 'hsv')
            
            data_pred_prim = data_pred.copy()
            
            for lbl in labels_pred:
                # crear imagen nifty con las segmentaciones en primarios secundarios y terciarios
                classification = lut_destrieux[lut_destrieux.todos == lbl].classification
                if len(classification) > 0: # primario o secundario
                    clas = classification.iloc[0]
                else: # terciario
                    clas = 3
                data_pred_prim[ data_pred_prim==lbl ] = clas

                # crear cortes por el centro de cada surco y guardarlos según su clasificacion y nombre 
                name = lut_destrieux[lut_destrieux.todos == lbl].name_todos
                if len(name) > 0: # primario o secundario
                    nombre = name.iloc[0]
                else: # terciario
                    nombre = 'terciario'
                f_name = str(clas) + '--' + str(nombre) + '--' + str(lbl)
                plot_sulcus(data_pred, lbl, affine, nib_brain, path_surcos, f_name, cmap)
                

            
            
if __name__ == '__main__':

    main()