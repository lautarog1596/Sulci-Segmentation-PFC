import os
from datetime import datetime
from matplotlib import pyplot as plt
import nibabel as nib
import medpy.metric.binary as metrics
import numpy as np
import pandas as pd
import seaborn as sns


def calc_dice_hd_classes(pred, target, labels, tmp_df):
    """ Por cada etiquta de pred y target, excepto la 0-background, calculo Dice y Hausdorff
    y devuelvo los valores en dos lista y las etiquetas de las segmentaciones calculadas"""

    keys = list(labels.keys())
    values = list(labels.values())

    for nro_lbl, nombre_etiqueta in zip(keys, values):
        bool_target_lbl = np.equal(target, nro_lbl)
        bool_pred_lbl = np.equal(pred, nro_lbl)

        dice = round(metrics.dc(bool_pred_lbl, bool_target_lbl), 4)
        hd = round(metrics.hd(bool_pred_lbl, bool_target_lbl), 4)

        new_row = {'etiqueta': nro_lbl, 'dice': dice, 'hausdorff': hd, 'nombre_etiqueta': nombre_etiqueta}
        tmp_df = tmp_df.append(new_row, ignore_index=True)

    return tmp_df



def main():
    # leer prediccion y target usado para entrenar, y calcular Dice y Hoausdorff
    # sobre toda la segmentacion y sobre las segmentaciones de cada clase

    # -----------------------
    # path to Datos de Test
    reg = 'MNINonLinear'
    path = '/media/DATOS/lgianinetto/HCP_datos/Test/'
    # namefile of target
    target_filename = 'bin_sul+sul_lbl_fus_geod_prop_remove_1.2_60_lbl100_defus.nii.gz'
    # csv file name output
    save_csv_filename = 'df_Dice_HD_M26.csv'
    save_csv_filename_classes = 'df_classes_Dice_HD_M26.csv'
    # namefile of prediction
    pred_filename1 = 'pred_M26_whole_defus_2.nii.gz'
    pred_model_filename = [pred_filename1]
    pred_model_name = ['M26']
    # se podría ver de la lut, pero por ahora los pongo a mano total no cambian
    labels = {1: 'lh_Lat_Fis', 2: 'lh_calcarine', 3: 'lh_central', 4: 'lh_parieto_occipital', 5: 'lh_pericallosal',
              6: 'lh_cingul-Marginalis',
              7: 'lh_circular_insula', 8: 'lh_collateral', 9: 'lh_front_inf', 10: 'lh_front_med', 11: 'lh_front_sup',
              12: 'lh_prim-Jensen',
              13: 'lh_intrapariet_and_P_trans', 14: 'lh_oc_middle_and_Lunatus', 15: 'lh_oc_sup_and_transversal',
              16: 'lh_occipital',
              17: 'lh_oc-temp_med_and_Lingual', 18: 'lh_orbital_lateral', 19: 'lh_orbital_med-olfact-H',
              20: 'lh_postcentral', 21: 'lh_precentral',
              22: 'lh_suborbital', 23: 'lh_subparietal', 24: 'lh_temporal', 25: 'rh_Lat_Fis', 26: 'rh_calcarine',
              27: 'rh_central',
              28: 'rh_parieto_occipital', 29: 'rh_pericallosal', 30: 'rh_cingul-Marginalis', 31: 'rh_circular_insula',
              32: 'rh_collateral', 33: 'rh_front_inf',
              34: 'rh_front_med', 35: 'rh_front_sup', 36: 'rh_prim-Jensen', 37: 'rh_intrapariet_and_P_trans',
              38: 'rh_oc_middle_and_Lunatus', 39: 'rh_oc_sup_and_transversal', 40: 'rh_occipital',
              41: 'rh_oc-temp_med_and_Lingual', 42: 'rh_orbital_lateral', 43: 'rh_orbital_med-olfact-H',
              44: 'rh_postcentral',
              45: 'rh_precentral', 46: 'rh_suborbital', 47: 'rh_subparietal', 48: 'rh_temporal', 100: 'terciarios'}
    # labels = {1:'secundarios', 2:'lh_Lat_Fis', 3:'lh_calcarine', 4:'lh_central', 5:'lh_parieto_occipital',
    # 6:'lh_pericallosal', 7: 'rh_Lat_Fis', 8: 'rh_calcarine', 9: 'rh_central',
    # 10: 'rh_parieto_occipital', 11: 'rh_pericallosal'}
    # -----------------------


    # Verificar que no haya otro csv guardado con dicho nombre
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile( cwd_path + '/result_accuracy/' + save_csv_filename ) or os.path.isfile( cwd_path + '/result_accuracy/' + save_csv_filename_classes ):
        print('Cargando archivos con estos nombres')
        df = pd.read_csv(cwd_path +'/result_accuracy/' + save_csv_filename, delimiter=',', decimal='.', encoding='utf-8')
        df_class = pd.read_csv(cwd_path + '/result_accuracy/' + save_csv_filename_classes, delimiter=',', decimal='.', encoding='utf-8')
    else:
        df_class = pd.DataFrame(columns=['model_name', 'id_paciente', 'etiqueta', 'dice', 'hausdorff', 'nombre_etiqueta'])
        df = pd.DataFrame(columns=['model_name', 'id_paciente', 'dice', 'x'])


    list_id_pacientes = [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]
    list_id_pacientes.sort()
    list_id_pacientes = list_id_pacientes[80:100]

    cant_pacientes = len(list_id_pacientes)

    ### Por c/paciente dentro de la carpeta fold
    for i, id_paciente in enumerate(list_id_pacientes, 1):
        print(f'- Paciente {id_paciente} [{i}/{cant_pacientes}] ... calculando accuracy')
        exists = df.isin([int(id_paciente)]).any().any()

        if not exists:
            # por cada modelo
            for model_filename, model_name in zip(pred_model_filename, pred_model_name):
                path1 = os.path.join(path, id_paciente, reg)
                path_target = os.path.join(path1, target_filename)
                path_pred = os.path.join(path1, model_filename)

                data_target = nib.load(path_target).get_fdata()
                data_pred = nib.load(path_pred).get_fdata()

                ### Evaluar accuracy
                # Sobre toda las segmentaciones
                dice = round(metrics.dc(data_pred!=0, data_target!=0), 4)
                tmp_df = pd.DataFrame({'model_name': [model_name], 'id_paciente': [id_paciente], 'dice': [dice], 'x' : [1]})
                df = df.append(tmp_df, ignore_index=True)
                # Sobre cada clase
                tmp_df = pd.DataFrame(columns = ['model_name','id_paciente', 'etiqueta', 'dice', 'hausdorff', 'nombre_etiqueta'])
                tmp_df = calc_dice_hd_classes(data_pred, data_target, labels, tmp_df)
                tmp_df['model_name'] = model_name
                tmp_df['id_paciente'] = id_paciente
                df_class = df_class.append(tmp_df, ignore_index=True)

                # guardar
                df.to_csv(cwd_path + '/result_accuracy/' + save_csv_filename, index=False)
                df_class.to_csv(cwd_path + '/result_accuracy/' + save_csv_filename_classes, index=False)


    ##### Plotear Dice sobre toda la segmentacion
    # Dice
    f1 = plt.figure(figsize=(6, 6))
    f1.suptitle('coef. Dice sobre la unión de las segmentaciones')
    sns.set_style("whitegrid")
    ax1 = sns.boxplot(x='x', y='dice', hue='model_name', data=df)
    ax1.set_ylim([0, 1])
    # Guardar
    now = datetime.now()
    nombre1 = 'boxplot_Dice_all_' + now.strftime("%Y%m%d-%H%M%S") + '.png'
    f1.savefig(cwd_path + '/result_accuracy/boxplots/' + nombre1)

    #### Plotear Dice y HD sobre cada clase segmentada
    # Dice
    f2 = plt.figure(figsize=(14, 8))
    plt.suptitle('coef. Dice por cada segmentación')
    ax = sns.boxplot(x='nombre_etiqueta', y='dice', hue='clasificacion', data=df_class)
    ax.set_ylim([0, 1])
    f2.autofmt_xdate()
    plt.tight_layout()
    # Hausdorff
    f3 = plt.figure(figsize=(14, 8))
    plt.suptitle('dist. HD por cada segmentación')
    sns.set_style("whitegrid")
    sns.boxplot(x="nombre_etiqueta", y="hausdorff", hue='clasificacion', data=df_class)
    f3.autofmt_xdate()
    plt.tight_layout()
    # Guardar
    now = datetime.now()
    nombre2 = 'boxplot_lbls_Dice_' + now.strftime("%Y%m%d-%H%M%S") + '.png'
    nombre3 = 'boxplot_lbls_Hausdorff_' + now.strftime("%Y%m%d-%H%M%S") + '.png'
    f2.savefig(cwd_path + '/result_accuracy/boxplots/' + nombre2)
    f3.savefig(cwd_path + '/result_accuracy/boxplots/' + nombre3)


if __name__ == '__main__':

    main()
    
    # prueba()


