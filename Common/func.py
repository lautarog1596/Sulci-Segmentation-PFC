import os
import shutil
from shutil import copyfile
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.plotting import view_img
from matplotlib import pyplot as plt
import numpy.linalg as npl


def create_dir(_path_out, *dirs):
    if not os.path.exists(_path_out):
        os.mkdir(_path_out)
    paths = []
    for dir in dirs:
        path_final = os.path.join(_path_out, dir)
        if not os.path.exists(path_final):
            os.mkdir(path_final)
        paths.append(path_final)
    return paths


def load_nibs(*paths):
    nibs = []
    for path in paths:
        nibs.append(nib.load(path))
    return nibs


def view_mri_web(nib_loaded):
    html_view = view_img(nib_loaded)
    html_view.open_in_browser()


def discard_rh(nib_pred):
    # Affine gives de map from voxel to scanner, and inverse of affine de scanner to voxel
    affine = nib_pred.affine
    affine_inv = npl.inv(affine)
    xyz_orig = [0, 0, 0]
    ijk_orig = nib.affines.apply_affine(affine_inv, xyz_orig).astype('int')

    # borro las etiquetas del emisferio derecho
    data = nib_pred.get_fdata().astype('int') # fundamental este astype('int')
    data[:ijk_orig[0], :, :] = 0
    nib_pred = nib.nifti1.Nifti1Image(data, affine, nib_pred.header)
    nib.save(nib_pred, '1.nii.gz')
    return nib_pred


def plot_slices1(nib_img, cut_xyz_coords, corte='z', back_ground=None, nib_img_contour=None):
    """
    :param nib_img: imagen de entrada formato nibabel
    :param cut_xyz_coords: coord x y z del corte central que luego se ploteará 8 cortes antes y después
    :param corte: x, y o z que indican los planos que se mostrarán
    :param back_ground: irm de fondo
    :return: display para poder guardar como imagen si se quiere
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle('Punto de corte: ' + str(cut_xyz_coords))

    if corte=='x':
        cuts = np.arange(cut_xyz_coords[0]-8, cut_xyz_coords[0]+8, 1)
    elif corte=='y':
        cuts = np.arange(cut_xyz_coords[1]-8, cut_xyz_coords[1]+8, 1)
    else:
        cuts = np.arange(cut_xyz_coords[2]-8, cut_xyz_coords[2]+8, 1)

    display = None
    for num, ax in enumerate(axes):
        cortes = cuts[ num*4 : 4+num*4 ]
        if back_ground is None:
            display = plotting.plot_roi(nib_img, display_mode=corte, axes=ax, cut_coords=cortes)
        else:
            display = plotting.plot_roi(nib_img, display_mode=corte, bg_img=back_ground, axes=ax, cut_coords=cortes)
        if nib_img_contour is not None:
            display.add_contours(nib_img_contour, filled=True, alpha=0.3, colors=('1', '1', '1'))
    # plt.show()

    return display

def plot_slices2(nib_img, cut_xyz_coords, back_ground=None, nib_img_contour=None, cmap=None, titulo='titulo'):
    """
    :param nib_img: imagen de entrada formato nibabel
    :param cut_xyz_coords: coord x y z del corte central que luego se ploteará 8 cortes antes y después
    :param corte: x, y o z que indican los planos que se mostrarán
    :param back_ground: irm de fondo
    :return: display para poder guardar como imagen si se quiere
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    cuts1 = np.arange(cut_xyz_coords[0]-4, cut_xyz_coords[0]+6, 2)
    cuts2 = np.arange(cut_xyz_coords[1]-4, cut_xyz_coords[1]+6, 2)
    cuts3 = np.arange(cut_xyz_coords[2]-4, cut_xyz_coords[2]+6, 2)
    cuts = [cuts1, cuts2, cuts3]
    dip_modes = ['x', 'y', 'z']
    ax_title = ['Sagital', 'Coronal', 'Axial']

    display = None
    for (dm, ax, cortes, tit) in zip(dip_modes, axes, cuts, ax_title):
        if back_ground is None:
            display = plotting.plot_roi(nib_img, display_mode=dm, axes=ax, cut_coords=cortes, cmap=cmap)
            ax.set_title(tit, size=16)
        else:
            display = plotting.plot_roi(nib_img, display_mode=dm, bg_img=back_ground, axes=ax, cut_coords=cortes, cmap=cmap)
            ax.set_title(tit, color='w', size=16)
        if nib_img_contour is not None:
            display.add_contours(nib_img_contour, filled=False, alpha=0.3, colors=('1', '1', '1'))

    # plt.show()
    if back_ground is not None:
        fig.suptitle(titulo, color='w', size=20)
    else:
        fig.suptitle(titulo, size=20)
    # fig.tight_layout()

    return display


#################

from nilearn.regions import connected_label_regions
# def descartar_comp_chicos(nib_image, threshold_vol):
def descartar_comp_chicos(nib_image, threshold_vol):
    """ threshold_vol, es en milimetros cúbicos.. eje. 100, equivale a 1cm cubico """
    # obtener data numpy
    numpy_data = nib_image.get_fdata()

    # volumen de un voxel
    vol_voxel = 1
    for pixdim in nib_image.header['pixdim'][1:4]:
        vol_voxel = vol_voxel * pixdim

    # listar volumenes en orden ascendente
    lbls = np.unique(numpy_data)[1:]  # no tengo en cuenta el 0 que es el fondo
    count_voxls = np.zeros_like(lbls).astype('int')  # volumenes de los surcos, usado para restringir la dilatacion
    for i, lbl in enumerate(lbls):
        bool_surc_destr = np.equal(numpy_data, lbl).astype('int')
        count_voxls[i] = np.count_nonzero(bool_surc_destr)
        # descartar componente si no cumple condicion
        if count_voxls[i] * vol_voxel < threshold_vol:
        # if count_voxls[i] < threshold_voxeles:
        #     print('etiqueta: ', lbl, ' eliminada')
            numpy_data[ bool_surc_destr==1 ] = 0

    # count_voxls.sort()
    # print('cantidad de voxeles: ', count_voxls)
    # print('volumenes: ', count_voxls * vol_voxel)
    return nib.nifti1.Nifti1Image(numpy_data, nib_image.affine)


#################

def recortar_mitad_cabeza(nib_img):
    # Affine gives de map from voxel to scanner, and inverse of affine de scanner to voxel
    affine = nib_img.affine
    affine_inv = npl.inv(affine)
    xyz_orig = [0, 0, 0]
    ijk_orig = nib.affines.apply_affine(affine_inv, xyz_orig).astype('int')

    # borro las etiquetas del emisferio derecho
    data = nib_img.get_fdata()
    new_left = data[ijk_orig[0]:, :, :]
    new_nib_left = nib.nifti1.Nifti1Image(new_left, nib_img.affine)

    return new_nib_left

#####################################

def remove_files():
    origin_path = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    folders = ['Test', 'Train', 'Val', 'Antonia']
    # folders = ['Test']
    reg = 'MNINonLinear'
    files_to_remove = ['bin_sul+sul_lbl_fus_geod_prop_remove_1.2_60_v2.nii.gz']

    list_list_origin_path, _, list_list_id_pacientes = list_of_path_files(origin_path, origin_path,
                                                                                            folders, reg)

    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path,
                                                                         list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)

        for i, (origin_path, id_paciente) in enumerate(
                zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            for file_to_copy in files_to_remove:
                path_orig = os.path.join(origin_path, file_to_copy)

                if not os.path.isfile(path_orig):
                    print('no se puede remover, no existe')
                else:
                    os.remove(path_orig)
                    print('removido')


def copy_file():
    origin_path = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    dest_path = '/home/lau/Escritorio/'
    # folders = ['Test', 'Train', 'Val']
    folders = ['Test']
    reg = 'T1w'
    files_to_copy = ['aparc.a2009s+aseg_primary_sulcus_labels.nii.gz']

    list_list_origin_path, list_list_dest_path, list_list_id_pacientes = list_of_path_files(origin_path, dest_path, folders, reg)
    
    # por cada carpeta
    for list_origin_path, list_dest_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_dest_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)
    
        for i, (origin_path, dest_path, id_paciente) in enumerate(zip(list_origin_path, list_dest_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            for file_to_copy in files_to_copy:
                path_orig = os.path.join(origin_path, file_to_copy)
                path_dest = os.path.join(dest_path, file_to_copy)

                if not os.path.isfile(path_dest):
                    copyfile(path_orig, path_dest)
                    print('copiado')
                else:
                    print('ya existe')


def move_file():
    # origin_path = '/media/lau/datos_facultad/datos_sinc_local/HCP_1200/enviar/'
    origin_path = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    dest_path = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/'
    names_orig_files = ['bin_sul+sul_lbl_fus_eroded_both_hemisf.nii.gz']
    names_moved_files = ['bin_sul+sul_lbl_fus_eroded_both_hemisf_v1.nii.gz']
    folders = ['Test', 'Train', 'Val', 'Antonia']
    # folders = ['Test']
    reg = 'MNINonLinear'
    
    list_list_origin_path, list_list_dest_path, list_list_id_pacientes = list_of_path_files(origin_path, dest_path, folders, reg)
    
    # por cada carpeta
    for list_origin_path, list_dest_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_dest_path, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)
    
        for i, (origin_path, dest_path, id_paciente) in enumerate(zip(list_origin_path, list_dest_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            for orig_f, moved_file in zip(names_orig_files, names_moved_files):
                path_orig = os.path.join(origin_path, orig_f)
                path_dest = os.path.join(dest_path, moved_file)

                if not os.path.isfile(path_dest):
                    shutil.move(path_orig, path_dest)
                    print('movido')
                else:
                    print('ya existe')


def list_of_path_files(orig_path, dest_path, folders, reg):
    """Retorna retorna lista con las rutas finales de los archivos de interes, más otra lista con los id de cada uno

    Args:
        path_base (string): Dirección origen donde están las carpetas Test, Train, Val que almacenan el archivo de interes
        folders (list): Lista con los nombres de las carpetas de interés: Test, Train, Val o Antonia
        reg (string): T1w o MNINonLinear, que son las sub carpetas posibles de cada paciente
    """    
    # recorrer la ruta: 'origin_path/fold/id_paciente/reg '
    # ej: '/media/lau/datos_facultad/pfc_final/Datos/Test/12123/T1w'
    
    list_list_origin_path = []
    list_list_dest_path = []
    list_list_id_pacientes = []
    for fold in folders:
        # print(f'\nDirecotrio: {fold}')
        path_fold_orig = os.path.join(orig_path, fold)
        path_fold_dest = os.path.join(dest_path, fold)
        if not os.path.exists(path_fold_dest):
            os.mkdir(path_fold_dest)

        list_id_pacientes = [id for id in os.listdir(path_fold_orig) if os.path.isdir(os.path.join(path_fold_orig, id))]
        list_id_pacientes.sort()
        list_list_id_pacientes.append(list_id_pacientes)

        list_path_fold_orig_id_reg = []
        list_path_fold_dest_id_reg = []
        cant_pacientes = len(list_id_pacientes)
        ### Por c/paciente dentro de la carpeta fold
        for i, id_paciente in enumerate(list_id_pacientes, 1):
            # print(f'Paciente {id_paciente} {i}/{cant_pacientes}')
            path_fold_orig_id = os.path.join(path_fold_orig, id_paciente)
            path_fold_dest_id = os.path.join(path_fold_dest, id_paciente)

            if not os.path.exists(path_fold_dest_id):
                os.mkdir(path_fold_dest_id)
            
            # carga atlas Destrieux
            path_fold_orig_id_reg = os.path.join(path_fold_orig_id, reg)
            path_fold_dest_id_reg = os.path.join(path_fold_dest_id, reg)
            if not os.path.exists(path_fold_dest_id_reg):
                os.mkdir(path_fold_dest_id_reg)
                
            list_path_fold_orig_id_reg.append(path_fold_orig_id_reg)
            list_path_fold_dest_id_reg.append(path_fold_dest_id_reg)
        list_list_origin_path.append(list_path_fold_orig_id_reg)
        list_list_dest_path.append(list_path_fold_dest_id_reg)

    return list_list_origin_path, list_list_dest_path, list_list_id_pacientes


def plot_hist_intens():
    import seaborn as sns
    nib_img = nib.load('/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/Antonia/102008/MNINonLinear/T1w_restore_brain_normZscores.nii.gz')
    nib_img2 = nib.load('/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/Antonia/102008/MNINonLinear/T1w_restore_brain.nii.gz')
    data = nib_img.get_fdata()
    data2 = nib_img2.get_fdata()
    data = data.flatten()
    data2 = data2.flatten()
    plt.rcParams.update({'font.size': 13})
    plt.figure()
    plt.title('Histograma de imagen\n normalizada con z-score')
    plt.xlabel('Valor de intensidad normalizado')
    plt.ylabel('Cantidad de vóxeles')
    sns.histplot(x=data, bins=10)
    plt.figure()
    plt.title('Histograma')
    plt.xlabel('Valor de intensidad')
    plt.ylabel('Cantidad de vóxeles')
    sns.histplot(data=data2, bins=10)


if __name__ == '__main__':
    a=0

    # #################
    # nib_destr = nib.load('/media/lau/datos_facultad/pfc_final/Datos/102008/aparc.a2009s+aseg_sulcus_labels.nii.gz')
    #
    # data_destr = nib_destr.get_fdata()
    #
    # lbls = np.unique(data_destr)
    # bool_data_destr_lbl = np.equal(data_destr, 12148).astype('int')
    #
    # # encontrar el centro de masa del surco
    # ijk_cut_coords = [ int(elem) for elem in center_of_mass(bool_data_destr_lbl)]
    # xyz_cut_coords = nib.affines.apply_affine(nib_destr.affine, ijk_cut_coords)
    #
    # # plot_slices1(nib_destr, xyz_cut_coords, 'x', back_ground=None, nib_img_contour=nib_destr)
    # plot_slices2(nib_destr, xyz_cut_coords, back_ground=None, nib_img_contour=nib_destr)
    # plt.show()
    # ########

    ################
    # descartar los componentes que no cumplan un determinado volúmen
    # descartar_comp_chicos()

    ################
    # nib_img = nib.load('/media/lau/datos_facultad/pfc_final/Datos/102008/T1w_restore_brain_normZscores.nii.gz')
    # new_nib_left = recortar_mitad_cabeza(nib_img)
    # nib.save(new_nib_left, '/media/lau/datos_facultad/pfc_final/Datos/102008/T1w_restore_brain_normZscores_' + 'left' + '.nii.gz')

    ###############
    # descartar hemisferio derecho de Destrieux
    # nib_destr = nib.load('/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/Antonia/102008/MNINonLinear/T1w_restore_brain_normZscores.nii.gz')
    # nib_destr_lh = discard_rh(nib_destr)
    # nib.save(nib_destr_lh, '/media/lau/datos_facultad/pfc_final/Datos/102008/aparc.a2009s+aseg_sulcus_labels_lh.nii.gz')

    ################
    # move_file()

    ################
    # remove_files()

    ################
    # copy_file()

    ################
    # plotear histograma intensidades imagen
    plot_hist_intens()



