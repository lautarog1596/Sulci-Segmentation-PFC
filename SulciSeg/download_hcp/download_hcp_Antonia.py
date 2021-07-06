# download_HCP_1200.py
#
# Author: Joke Durnez (joke.durnez@gmail.com)

'''
This script downloads data from the Human Connectome Project - 1200 subjects release.
'''

# Import packages
import fnmatch
import os
import pandas
import boto3

from Common.func import list_of_path_files


def set_up_connection(path_credentials):
    """ Establecer coneccion con amazon AWS una sola vez """
    credentials = pandas.read_csv(path_credentials)
    params = dict(
        bucket='hcp-openaccess',
        out_path='../../Datos',
        aws_key=credentials['Access Key Id'].iloc[0],
        aws_secret=credentials['Secret Access Key'].iloc[0],
        overwrite=False,
        prefix='HCP_1200')
    return params


# Main collect and download function
def collect_and_download(archivos, out_dir, subject, reg, params):

    # Init variables
    s3_bucket_name = 'hcp-openaccess'
    s3_prefix = 'HCP_1200'

    s3 = boto3.resource('s3', aws_access_key_id=params['aws_key'], aws_secret_access_key=params['aws_secret'],)
    bucket = s3.Bucket(s3_bucket_name)

    s3_keys = bucket.objects.filter(Prefix='HCP_1200/'+subject+'/'+reg)


    # me quedo solo con las que me interesan
    s3_keylist= [key.key for key in s3_keys
                    for archivo in archivos if fnmatch.fnmatch(key.key, '*/'+subject+'.'+archivo)
                 ]

    # If output path doesn't exist, create it
    if not os.path.exists(out_dir):
        print(f'Could not find {out_dir}, creating now...')
        os.makedirs(out_dir)

    # Init a list to store paths.
    # print('Collecting images of interest...')

    # And download the items
    total_num_files = len(s3_keylist)
    files_downloaded = len(s3_keylist)
    for path_idx, s3_path in enumerate(s3_keylist):
        rel_path = s3_path.replace(s3_prefix, '')
        rel_path = rel_path.lstrip('/')
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        try:
            if not os.path.exists(download_file):
                print(f'Downloading to: {download_file}')
                with open(download_file, 'wb') as f:
                    bucket.download_file(s3_path,download_file)
                # print("FACTS: path: %s, file: %s"%(s3_path, download_file))
                # print(f'{(100*(float(path_idx+1)/total_num_files))} percent complete')
            else:
                print(f'File {download_file} already exists, skipping...')
                files_downloaded -= 1
        except Exception as exc:
            print('There was a problem downloading {s3_path}.\n'
                  'Check and try again.')
            print(exc)


    print(f'{files_downloaded} files downloaded for subject {subject}.')

    print('Done!')


def main():
    # direccion donde estan los sujetos almacenados
    path_subjects = '/media/lau/datos_facultad/datos_sinc_local/HCP_1200/'
    # folders = ['Antonia', 'Test', 'Train', 'Val']
    folders = ['Antonia']
    reg = 'MNINonLinear'
    path_credentials = '/home/lau/Escritorio/AWS_key_HCP.csv'

    archivos = [
        # 'T1w_acpc_dc_restore_brain.nii.gz',
        # 'aparc.a2009s+aseg.nii.gz',
        'ribbon.nii.gz',
        # 'L.sulc.32k_fs_LR.shape.gii',
        # 'L.midthickness.32k_fs_LR.surf.gii',
        # 'L.white.32k_fs_LR.surf.gii',
        # 'L.pial.32k_fs_LR.surf.gii'
    ]

    ### ----------------------------------------------

    # establecer coneccion con amazon
    params = set_up_connection(path_credentials)

    # listo todos los sujetos

    list_list_origin_path, _, list_list_id_pacientes = list_of_path_files(path_subjects, path_subjects, folders, reg)

    # por cada carpeta
    for list_origin_path, list_id_pacientes, fold in zip(list_list_origin_path, list_list_id_pacientes, folders):
        list_origin_path = list_origin_path[:1]
        list_id_pacientes = list_id_pacientes[:1]

        print(f'---- Folder {fold}')
        cant_pacientes = len(list_origin_path)

        for i, (origin_path, id_paciente) in enumerate(zip(list_origin_path, list_id_pacientes)):
            print(f'-Paciente [{i}/{cant_pacientes}]: {id_paciente}')

            collect_and_download(archivos, out_dir=origin_path, subject=id_paciente, reg=reg, params=params)



if __name__ == '__main__':

    main()