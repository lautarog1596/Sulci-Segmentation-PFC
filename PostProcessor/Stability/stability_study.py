import os
import numpy as np
import nibabel as nib
import seaborn as sns
from nilearn.regions import connected_label_regions
from matplotlib import pyplot as plt
import pandas as pd


def calc_mapas_prob_var(save_path, reg, path_test, name_file, labels):
    """ Por cada surco, calcular un mapa de probabilidad con todas sus segmentaciones predichas de Test
        También calcular mapa de varianza, esto dá los lugares donde hay más acuerdo y disparidad entre las salidas"""

    list_id_pacientes = [filename for filename in os.listdir(path_test) if os.path.isdir(os.path.join(path_test, filename))]
    # list_id_pacientes = list_id_pacientes[:10]

    cant_pacientes = len(list_id_pacientes)

    ###### Cálculo de mapa de probabilidad
    # lista donde se guardará la matriz de probabilidad de cada una de las etiquetas
    list_data_prob_map = [None] * len(labels)
    affine = None
    for i, id_paciente in enumerate(list_id_pacientes, 1):
        print(f'- Paciente {id_paciente} [{i}/{cant_pacientes}] ... calculando mapa de probabilidad')
        # cargar imagen
        path_pred = os.path.join(path_test, id_paciente, reg, name_file)
        nib_pred = nib.load(path_pred)
        if affine is None:
            affine = nib_pred.affine
        data_pred = nib_pred.get_fdata()

        # acumular para al final obtener la media o probabilidad
        for i, lbl in enumerate(labels):
            if list_data_prob_map[i] is None:
                list_data_prob_map[i] = np.zeros_like(data_pred).astype('uint8')
            bool_data_pred_lbl = np.equal(data_pred, lbl).astype('uint8')
            list_data_prob_map[i] += bool_data_pred_lbl

    # obtención de media o probabilidad
    list_data_prob_map = np.array(list_data_prob_map) / cant_pacientes
    # guardar los mapas de probabilidades
    print('Guardando los mapas calculados')
    for lbl, data in zip(labels, list_data_prob_map):
        nib_pred_cc = nib.nifti1.Nifti1Image(data, affine)
        nib.save(nib_pred_cc, os.path.join(save_path, 'mapa_probab_lbl' + str(lbl) + '.nii.gz'))

    # # ------------------------------------------
    #
    # ###### Cálculo del mapa de varianza
    # list_data_var_map = [None] * len(labels)
    # for i, id_paciente in enumerate(list_id_pacientes, 1):
    #     print(f'- Paciente {id_paciente} [{i}/{cant_pacientes}] ... calculando mapa de varianza')
    #     # cargar imagen
    #     path_pred = os.path.join(path_test, id_paciente, reg, name_file)
    #     data_pred = nib.load(path_pred).get_fdata()
    #
    #     # calculo de varianza
    #     for i, lbl in enumerate(labels):
    #         if list_data_var_map[i] is None:
    #             list_data_var_map[i] = np.zeros_like(data_pred).astype('float64')
    #         bool_data_pred_lbl = np.equal(data_pred, lbl).astype('int8').astype('float64')
    #         list_data_var_map[i] += np.power(bool_data_pred_lbl - list_data_prob_map[i], 2)
    #
    # # obtención de desviación típica o raiz cuadrada de la varianza
    # list_data_var_map = np.sqrt( np.array(list_data_var_map) / cant_pacientes )
    # # guardar los mapas de probabilidades
    # print('Guardando los mapas calculados')
    # for lbl, data in zip(labels, list_data_var_map):
    #     nib_pred_cc = nib.nifti1.Nifti1Image(data, affine)
    #     nib.save(nib_pred_cc, os.path.join(save_path, 'mapa_var_lbl' + str(lbl) + '.nii.gz'))

def est(data_pred, data_gt, union):
# def est(data_pred, union):
    # regiones donde esta el surco en la prediccion y en el GT
    bool_data_pred = data_pred > union
    bool_data_gt = data_gt > union
    
    # union entre ambas regiones
    bool_data_all = bool_data_pred | bool_data_gt
    # bool_data_all = bool_data_pred

    # cantidad de voxeles de la región
    cant_voxls_union = np.sum(bool_data_all)
    
    # suma de probabilidad mayor a 0, de la prediccion y del GT
    sum_prob_pred = np.sum(data_pred[bool_data_all])
    sum_prob_gt = np.sum(data_gt[bool_data_all])

    # cálculo de medias
    estabilidad_pred = round(sum_prob_pred/cant_voxls_union, 3) 
    estabilidad_gt = round(sum_prob_gt/cant_voxls_union, 3)
    
    # cálculo de desvíos estándar
    std_pred = round(np.std(data_pred[bool_data_all]), 3)
    std_gt = round(np.std(data_gt[bool_data_all]), 3)

    return estabilidad_pred, std_pred, estabilidad_gt, std_gt
    # return estabilidad_pred, std_pred


    
def calc_estabilidad(save_path, reg, labels, titulos, union):
    """ Se calcula sobre la unión de todos los valores positivos de los mapas prediccion y GT del surco.
    el valor reg indica si calcular sobre las registradas linealmente o no"""
    # nombre del mapa
    prob_map_filename = 'mapa_probab_lbl' # ejemplo: mapa_probab_lbl7.nii.gz
    save_path = os.path.dirname(save_path)
    
    if reg == 'T1w':
        path_pred = os.path.join(save_path, 'pred_linear')
        path_gt = os.path.join(save_path, 'gt_linear')
    else:
        path_pred = os.path.join(save_path, 'pred_non_linear')
        path_gt = os.path.join(save_path, 'gt_non_linear')

    # obtener estabilidad por cada mapa de probabilidad
    estabilidades_pred = []
    estabilidades_gt = []
    stds_pred = []
    stds_gt = []
    for lbl in labels:
        # cargar la prediccion y el GT del surco
        path_pred1 = os.path.join(path_pred, prob_map_filename + str(lbl) + '.nii.gz' )
        path_gt1 = os.path.join(path_gt, prob_map_filename + str(lbl) + '.nii.gz' )
        data_pred = nib.load(path_pred1).get_fdata()
        data_gt = nib.load(path_gt1).get_fdata()
        
        # calcular estabilidad y desvio estandar
        estabilidad_pred, std_pred, estabilidad_gt, std_gt = est(data_pred, data_gt, union)
        # estabilidad_pred, std_pred = est(data_pred, union)

        # guardar los resultados
        estabilidades_pred.append(estabilidad_pred)
        estabilidades_gt.append(estabilidad_gt)
        stds_pred.append(std_pred)
        stds_gt.append(std_gt)

        print(f' Estabilidad surco con etiqueta {lbl}:   prediccion: {estabilidad_pred}+-{std_pred},  GT: {estabilidad_gt}+-{std_gt}')
        # print(f' Estabilidad surco con etiqueta {lbl}:   prediccion: {estabilidad_pred}+-{std_pred}')

    # guardar en csv
    dict = {'label': labels, 'name': titulos, 'estabilidad_pred': estabilidades_pred, 'std_pred': stds_pred, 'estabilida_gt': estabilidades_gt, 'std_gt': stds_gt}
    # dict = {'label': labels, 'name': titulos, 'estabilidad_pred': estabilidades_pred, 'std_pred': stds_pred}
    df = pd.DataFrame(dict)
    nombre_archivo = 'valores_estabilidad_'+ reg + '_' + str(union) +'.csv'
    df.to_csv(os.path.join(save_path, nombre_archivo), index=False)
    print('Valores guardados en: ', os.path.join(save_path, nombre_archivo))


def graficar_mapas_3d(save_path, labels, titulos):
    from nilearn import plotting

    for lbl, tit in zip(labels, titulos):
        if tit=='lh_calcarine' or tit=='lh_central' or tit=='lh_pericallosal' or \
                tit=='lh_prim-Jensen' or tit=='lh_orbital_lateral':
            path_map = os.path.join(save_path, 'mapa_probab_lbl' + str(lbl) + '.nii.gz')

            view = plotting.view_img_on_surf(path_map, surf_mesh='fsaverage', threshold=0.0001, vmax=1)
            view.open_in_browser()


def plot_hist_mapa_probab(path_save, reg, labels, titulos, union):
    # nombre del mapa
    prob_map_filename = 'mapa_probab_lbl'  # ejemplo: mapa_probab_lbl7.nii.gz
    path_save = os.path.dirname(path_save)

    if reg == 'T1w':
        path_pred = os.path.join(path_save, 'pred_linear')
        path_gt = os.path.join(path_save, 'gt_linear')
    else:
        path_pred = os.path.join(path_save, 'pred_non_linear')
        path_gt = os.path.join(path_save, 'gt_non_linear')

    # por cada mapa de probabilidad, por cada surco, obtengo histograma
    estabilidades_pred = []
    estabilidades_gt = []
    for lbl, tit in zip(labels, titulos):
        # cargar la prediccion y el GT del surco
        path_pred1 = os.path.join(path_pred, prob_map_filename + str(lbl) + '.nii.gz')
        path_gt1 = os.path.join(path_gt, prob_map_filename + str(lbl) + '.nii.gz')
        data_pred = nib.load(path_pred1).get_fdata()
        data_gt = nib.load(path_gt1).get_fdata()

        # regiones donde esta el surco en la prediccion y en el GT
        bool_data_pred = data_pred > union
        bool_data_gt = data_gt > union

        # union entre ambas regiones
        bool_data_all = bool_data_pred | bool_data_gt
        # bool_data_all = bool_data_pred

        # probabilidades de la región
        probabs_pred = data_pred[bool_data_all].flatten()
        probabs_gt = data_gt[bool_data_all].flatten()
        probabs = np.concatenate((probabs_pred, probabs_gt), axis=None)
        # probabs = probabs_pred

        # obtener área bajo la curva a partir de 0.5 de probabilidad
        bins = np.arange(0, 1, 0.01)
        values_pred, bins = np.histogram(probabs_pred, bins=bins)
        values_gt, bins = np.histogram(probabs_gt, bins=bins)
        values_pred = values_pred / np.count_nonzero(bool_data_all)
        values_gt = values_gt / np.count_nonzero(bool_data_all)
        area_segunda_mitad_pred = sum(values_pred[len(values_pred) // 2:])
        area_segunda_mitad_gt = sum(values_gt[len(values_pred) // 2:])
        # area_total == area_primer_mitad + area_segunda_mitad == 1

        # crear DataFrame para plotear
        nombre_pred = ['Predicción'] * len(probabs_pred)
        nombre_gt = ['GT'] * len(probabs_gt)
        nombres = nombre_pred + nombre_gt
        # nombres = nombre_pred
        datos = {'': nombres, 'probabs': probabs}
        df = pd.DataFrame(datos)

        # medias
        mean_pred = probabs_pred.mean()
        mean_gt = probabs_gt.mean()

        # # graficar histograma
        # ax = sns.displot(df, x='probabs', hue='', kde=True)
        # plt.figure()
        # ax = sns.histplot(df, x='probabs', hue='', stat='probability', bins=bins, common_norm=False)
        # line = ax.axvline(mean_pred, color='b', linestyle='--')
        # line.set_label('Media - Predicción')
        # line = ax.axvline(mean_gt, color='darkorange', linestyle='--')
        # line.set_label('Media - GT')
        # plt.xlim(-0.05, 1.05)
        # plt.legend()
        # plt.xlabel('Probabilidad')
        # plt.ylabel('Cantidad de vóxeles normalizada')
        # plt.title('Histograma normalizado surco: ' + tit +
        #           '\n\nEstabilidad predicción: ' + str(round(area_segunda_mitad_pred, 4)) +
        #           '\nEstabilidad GT: ' + str(round(area_segunda_mitad_gt, 4)))
        # # plt.title('Histograma normalizado surco: ' + tit +
        # #           '\n\nEstabilidad predicción: ' + str(round(area_segunda_mitad_pred, 4)) )
        # plt.tight_layout()
        #
        # # guardar figura
        # plt.savefig(os.path.join(path_save, 'histogramas', reg + '_lbl_' + str(lbl) + '.png'))
        # print(f'Histograma {reg} {lbl} guardado')
        # plt.close()
        # # plt.show()

        # guardar los resultados
        estabilidades_pred.append(round(area_segunda_mitad_pred, 4))
        estabilidades_gt.append(round(area_segunda_mitad_gt, 4))

    # guardar en csv
    dict = {'label': labels, 'name': titulos, 'estabilidad_pred': estabilidades_pred, 'estabilida_gt': estabilidades_gt}
    # dict = {'label': labels, 'name': titulos, 'estabilidad_pred': estabilidades_pred}
    df = pd.DataFrame(dict)
    nombre_archivo = 'valores_estabilidad_hist_' + reg + '_' + str(union) + '.csv'
    df.to_csv(os.path.join(path_save, nombre_archivo), index=False)
    print('Valores guardados en: ', os.path.join(path_save, nombre_archivo))


def crear_directorio(reg, pred, save_path):

    # crear directorio donde guardar si no existe
    if reg == 'T1w' and pred == True:
        save_path = os.path.join(os.getcwd(), save_path, 'pred_linear')
    elif reg == 'T1w' and pred == False:
        save_path = os.path.join(os.getcwd(), save_path, 'gt_linear')
    elif reg == 'MNINonLinear' and pred == True:
        save_path = os.path.join(os.getcwd(), save_path, 'pred_non_linear')
    elif reg == 'MNINonLinear' and pred == False:
        save_path = os.path.join(os.getcwd(), save_path, 'gt_non_linear')
    else:
        print('errorrrrr')
        exit()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path


def obtener_labels_titulos_from_lut(path_lut):
    lut = pd.read_csv(path_lut, delimiter=',')

    labels = lut['etiqueta_sim']  # todas las etiquetas de los primarios, que son del 1 al 11
    titulos = lut['nom_sim']
    indx_lbl_ordered = np.argsort(labels)  # orden de las etiquetas de menor a mayor
    temp_labels = []  # lista que no tiene etiquetas repetidas
    temp_titulos = []  # lista que no tiene etiquetas repetidas
    for lbl, titulo in zip(labels[indx_lbl_ordered], titulos[indx_lbl_ordered]):
        if lbl not in temp_labels:
            temp_labels.append(lbl)
            temp_titulos.append(titulo)
    labels = temp_labels
    titulos = temp_titulos
    return labels, titulos


def main():

    # ----------- parametros
    path_test = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/Test/'
    # registracion lineal o no
    reg = 'T1w'  # T1w o MNINonLinear
    # si pred es true, se calcula la estabilidad sobre las predicciones, si es false sobre GT
    pred = False
    # porcentaje de descarte union
    union = 0.2
    # nombre del archivo predicho o gt, sobre los cuales se hará el analisis de estabilidad
    # name_file = 'aparc.a2009s+aseg_sulcus_labels_re_etiq.nii.gz' # GT no_lineal
    # name_file = 'aparc.a2009s+aseg_sulcus_labels_re_etiq.nii.gz' # GT lineal
    # name_file = 'pred_M25_whole_defus.nii.gz' # pred MNINonLinear
    name_file = 'pred_M25_lineal_whole_defus.nii.gz'  # pred T1w
    save_path = 'resultados'  # pred_non_linear, gt_non_linear, pred_non_linear, gt_linear
    save_path = crear_directorio(reg, pred, save_path)
    path_lut = '/media/lau/datos_facultad/PFC/Codigo/pfc_final/Datos/LUT_Destrieux.csv'
    labels, titulos = obtener_labels_titulos_from_lut(path_lut)
    # -----------

    ### Generar las imágenes de los mapas de probabilidad y varianza por cada surco primario.
    # direccion donde estan los datos de test
    # calc_mapas_prob_var(save_path, reg, path_test, name_file, labels)

    ### Calcular el valor de estabilidad de cada surco primario. Ej. central: 0.32
    #calc_estabilidad(save_path, reg, labels, titulos, union)

    ### Obtener histograma por cada mapa de estabilidad de cada surco
    #plot_hist_mapa_probab(save_path, reg, labels, titulos, union)  # usar este

    ### Graficar los mapas más lindos en 3d
    graficar_mapas_3d(save_path, labels, titulos)




if __name__ == '__main__':

    main()



