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
def collect_and_download(out_dir, subject, params):

    # Init variables
    s3_bucket_name = 'hcp-openaccess'
    s3_prefix = 'HCP_1200'

    s3 = boto3.resource('s3', aws_access_key_id=params['aws_key'], aws_secret_access_key=params['aws_secret'],)
    bucket = s3.Bucket(s3_bucket_name)

    s3_keys = bucket.objects.filter(Prefix='HCP_1200/%s/T1w'%subject)

    # me quedo solo con las que me interesan
    s3_keylist= [key.key for key in s3_keys
                    if (
                        fnmatch.fnmatch(key.key, '*/T1w_acpc_dc_restore_brain.nii.gz')
                        or fnmatch.fnmatch(key.key, '*/aparc.a2009s+aseg.nii.gz')
                    )]

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
    path_subjcts = '/media/lau/datos_facultad/pfc/Datos/Train'
    path_credentials = '/preprocesamiento/download_hcp/aws_hcp_details.csv'

    ### ----------------------------------------------

    # establecer coneccion con amazon
    params = set_up_connection(path_credentials)

    # listo todos los sujetos
    list_id_pacientes = [filename for filename in os.listdir(path_subjcts)
                         if os.path.isdir(os.path.join(path_subjcts, filename))]
    list_id_pacientes.sort()
    list_id_pacientes = list_id_pacientes[160:161]
    cant_pacientes = len(list_id_pacientes)

    ### Por c/paciente dentro de la carpeta path_subjcts, descargo las imagenes que quiero
    for i, id_paciente in enumerate(list_id_pacientes, 1):
        print(f'- Paciente {id_paciente} [{i}/{cant_pacientes}]')

        collect_and_download(out_dir=path_subjcts, subject=id_paciente, params=params)


if __name__ == '__main__':

    main()