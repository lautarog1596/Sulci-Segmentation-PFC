import os
from PreProcessor.join_lbl import reconstr_geod
import nibabel as nib
from Common.func import load_nibs
from Common.func import discard_rh, list_of_path_files



def create_GT(path_imgs, filename_destr, filename_bin, folders, reg, lbl, filename_out, only_left,
              restriccion_vol, borrar_chicos):

    list_list_paths, _, list_list_id_pacientes = list_of_path_files(path_imgs, path_imgs, folders, reg)

    # por cada carpeta
    for list_paths, list_id_pacientes, fold in zip(list_list_paths, list_list_id_pacientes, folders):
        print(f'---- Folder {fold}')

        list_id_pacientes.sort()
        cant_pacientes = len(list_id_pacientes)

        for i, (path, id_paciente) in enumerate(zip(list_paths, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            left = ''
            if only_left:
                left = '_lh'

            path_destr = os.path.join(path, filename_destr)
            path_bin = os.path.join(path, filename_bin)
            path_out = os.path.join(path, filename_out)
            path_out = path_out + str(restriccion_vol) + '_' + str(borrar_chicos) + left + '.nii.gz'

            if not os.path.exists(path_out):
                nib_bin, nib_destr = load_nibs(path_bin, path_destr)

                ### Identificar los surcos Destrieux encontrados, clasificarlos
                print('- Segmentando instancias mediante reconstrucción geodésica')

                if only_left:
                    # tomar etiquetas sólo del hemisferio izquierdo
                    nib2 = discard_rh(nib_bin)

                nib_inst_segm = reconstr_geod(nib_bin, lbl, nib_destr, restriccion_vol=restriccion_vol,
                                              borrar_chicos=borrar_chicos, erode=False)
                nib.save(nib_inst_segm, path_out)





def main():
    """ Este script toma la imagen binaria de sulsicidad y le agrega las etiquetas de Destrieux.
        1- sustituye las etiquetas de Destrieux en la imagen de sulsicidad.
        2- dilata las etiquetas restringiendo a los voxeles que poseen etiqueta de sulsicidad y al 20% de la cant.
        total de vóxeles de dicho surco.
        3- elimina las segmentaciones de sulsicidad que son menores a un umbral
    """

    # ----------- parámetros
    restriccion_vol = 1.2   # valor para restringir la propagacion de la etiqueta destrieux sobre la de sulsicidad.
    borrar_chicos = 60    # umbral en mm cubicos para borrar segmentaciones de sulsicidad menores. 100 sería 1cm cubico
    # -----------

    # ----------- mas parametros
    # ruta base de las imágenes
    path_imgs = '/media/lau/datos_facultad/pfc_final/Datos'
    # carpetas en dodne estan los archivos: Test, Train, Val, Antonia
    folders = ['Test']
    reg = 'MNINonLinear'
    # nonmbre de la imagen de la cual sustituir sus etiquetas
    filename_destr = 'aparc.a2009s+aseg_sulcus_labels_fusionado.nii.gz'
    # nombre de la imagen binaria a sustituir
    filename_bin = 'sulcus.ribbon_binSulc_25lbl.nii.gz'
    lbl = 25 # etiqueta que tiene esta segmentacion binaria
    # nombre del archivo segmentado
    filename_out = 'bin_sul+sul_lbl_fus_geod_prop_remove_'  ############ cambiar según sea train1 train2 ....
    # segmentar solo izquierdo o todo cerebro
    only_left = False # esto es para poder ver después los resultados, xq los dos hemisferios en 3d se complica
    # -----------

    
    create_GT(path_imgs, filename_destr, filename_bin, folders, reg, lbl, filename_out, only_left, restriccion_vol, borrar_chicos)





if __name__ == '__main__':

    main()




