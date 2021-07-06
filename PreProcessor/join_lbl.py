import numpy as np
import skimage.morphology as morph
import nibabel as nib
from nilearn.regions import connected_label_regions
from Common.func import descartar_comp_chicos


def add_Destr2bin(pred_and_destr_transf, data_destr):
    # transferir las etiq de Destr que le faltan a la binaria, para hacer la union de etiquetas
    pos = (pred_and_destr_transf==0) & (data_destr!=0)
    pred_and_destr_transf[pos] = data_destr[pos]
    return pred_and_destr_transf

def pred_and_destrieux(data_pred, lbl_pred, data_destr):
    pred_and_destr = np.copy(data_pred)
    cond = (data_pred == lbl_pred) & (data_destr != 0)
    pred_and_destr[cond] = data_destr[cond]
    return pred_and_destr


def reconstr_geod(nib_bin, lbl_pred, nib_destr, restriccion_vol = 0, borrar_chicos=50, erode = False):
    """ si restriccion_vol == 0 no hay restriccion.. si vale 1.2 por ej. no se propaga la etiqueta de Destrieux
     mas del 20% del volumen del surco de Destrieux"""

    # obtener data numpy
    data_destr = nib_destr.get_fdata().astype('int16')
    data_pred = nib_bin.get_fdata().astype('int16')

    # imagen predicha con las etiquetas de destrieux en donde se superponen
    reconstruida = pred_and_destrieux(data_pred, lbl_pred, data_destr).astype('int16')

    # agregarle a la imagen anterior lo que falta de Destrieux. (como superponer Destrieux sobre la predicha binaria)
    reconstruida = add_Destr2bin(reconstruida, data_destr)

    reconstr_ant = np.zeros_like(reconstruida)

    # lista con las etiquetas de todos los surcos de destrieux
    lbls_destr = np.unique(data_destr)[1:] # no tengo en cuenta el 0 que es el fondo
    imgs_surcos = [] # lista de imágenes binarias de cada surco de Destrieux
    volumenes_surcos = list(np.zeros_like(lbls_destr)) # volumenes de los surcos, usado para restringir la dilatacion
    for i, lbl in enumerate(lbls_destr):
        bool_surc_destr = np.equal(data_destr, lbl).astype('uint16')
        imgs_surcos.append(bool_surc_destr)
        if restriccion_vol != 0:
            vol = np.count_nonzero(bool_surc_destr)
            volumenes_surcos[i] = vol

    selem = morph.ball(2)
    while (reconstruida != reconstr_ant).any():

        reconstr_ant = reconstruida.copy()

        quitar_surcos = [] # lista donde se irán guardando los surcos que ya se dilataron por completo
        for i, (lbl, img_surco, volumen) in enumerate(zip(lbls_destr, imgs_surcos, volumenes_surcos)):
            reconstr_ant2 = reconstruida.copy()

            bool_actual_lbl = reconstruida==lbl
            if np.count_nonzero(bool_actual_lbl) < volumen * restriccion_vol or restriccion_vol == 0: # seguir dilatando si se puede

                # mask_image es la que restringirá la dilatación y se modificará contínuamente
                mask_image = np.bitwise_or(reconstruida==lbl_pred, bool_actual_lbl).astype('int16') # voxeles que faltan reconstruír (valor = lbl_pred) y de la etiqueta actual
                # crecer region restringido a donde falta crecer
                img_surco_dilatado = np.bitwise_and( morph.binary_dilation(img_surco, selem), mask_image )
                imgs_surcos[i] = img_surco_dilatado
                reconstruida[ np.where(img_surco_dilatado==1) ] = lbl

            # si reconstruida es igual a la anterior, quito dicho surco y lbl de la lista xq ya se dilataron al max
            if (reconstruida == reconstr_ant2).all():
                quitar_surcos.append(i)

        if len(quitar_surcos)!=0:
            lbls_destr = [x for i, x in enumerate(lbls_destr) if i not in quitar_surcos]
            imgs_surcos = [x for i, x in enumerate(imgs_surcos) if i not in quitar_surcos]
            volumenes_surcos = [x for i, x in enumerate(volumenes_surcos) if i not in quitar_surcos]
            # print('quedan:', lbls_destr)

    if borrar_chicos!=-1:
        # separar en dos imagenes, una solo terciarios y otra el resto
        bool_solo_terc = reconstruida == lbl_pred
        data_sin_terciarios = reconstruida.copy()
        data_sin_terciarios[data_sin_terciarios == lbl_pred] = 0

        nib_solo_terc = nib.nifti1.Nifti1Image(bool_solo_terc.astype('uint8')*lbl_pred, nib_bin.affine)

        # componentes conectadas
        nib_solo_terc_cc = connected_label_regions(nib_solo_terc, connect_diag=True)
        nib_solo_terc_cc = nib.nifti1.Nifti1Image(nib_solo_terc_cc.get_fdata(), nib_bin.affine)

        # descartar surcos pequeños
        nib_solo_terc_cc = descartar_comp_chicos(nib_solo_terc_cc, threshold_vol=borrar_chicos)
        data_solo_terc_cc = nib_solo_terc_cc.get_fdata()
        data_solo_terc_cc[data_solo_terc_cc!=0] = lbl_pred
        temp = data_solo_terc_cc

        reconstruida = data_sin_terciarios + temp

    nib_reconstruida = nib.nifti1.Nifti1Image(reconstruida, nib_bin.affine)
    return nib_reconstruida



if __name__ == '__main__':
    pass


