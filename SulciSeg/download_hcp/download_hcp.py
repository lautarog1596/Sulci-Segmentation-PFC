#!/usr/bin/env python

import os
import os.path as op
import glob
import re
import fnmatch
import signal
import logging

# import boto3
import boto
import pandas
from joblib import Parallel, delayed

credential_path = op.expanduser('aws_hcp_details.csv')
# HCP_FOLDER = "HCP_1200"
HCP_FOLDER = "/media/lau/datos_facultad/pfc/Datos/HCP_download"
N_JOBS = 1

credentials = pandas.read_csv(credential_path)
# subjects_there = set( # sets permite que no haya elementos repetidos
#     int(re.findall('[0-9]+$', s)[0]) for s in glob.glob(op.join(HCP_FOLDER, '[0-9]*[0-9]'))
#     # [0-9] que contenga caracteres en ese rango, $ indica que terminen en ese rango, + uno o mas caracteres del rango
# )
#
# subjects = list(subjects_there)
# subjects.sort()
# subjects = subjects[:1]
# print("Total subjects {}".format(len(subjects)))

max_runs = 3
run_inds = [0, 1, 2]
n_jobs = 24

# hcp_data_types = [
#     'rest',
# ]
# hcp_outputs = [
#     'diffusion'
# ]
# hcp_onsets = ['', '']
# hcp_suffix = 'HCP_1200'
params = dict(
    bucket='hcp-openaccess',
    out_path='../..',
    aws_key=credentials['Access Key Id'].iloc[0],
    aws_secret=credentials['Secret Access Key'].iloc[0],
    overwrite=False,
    prefix='HCP_1200')

# hcp_prefix = '{prefix}'.format(**params)
# hcp_prefix


s3 = boto.connect_s3(
    aws_access_key_id=params['aws_key'],
    aws_secret_access_key=params['aws_secret']
)
# s3 = boto3.client(
#     's3',
#     aws_access_key_id=params['aws_key'],
#     aws_secret_access_key=params['aws_secret'],
# )
# s3 = session.resource('s3')

s3.debug = 0
s3_bucket = s3.get_all_buckets()[0]
# s3_bucket = [bucket for i, bucket in enumerate(s3.buckets.all(), 0) if i==1][0]


class DelayedSignals(object):
    signals_to_handle = [
        signal.SIGINT,
        signal.SIGKILL,
        signal.SIGTERM
    ]

    def __enter__(self):
        self.signal_received = False
        self.old_handlers = {
            s: signal.signal(s, self.handler)
            for s in self.signals_to_handle
        }
        self.signals_received = []

    def handler(self, sig, frame):
        self.signal_received.append((sig, frame))
        logging.debug('Signal received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        for k, v in self.old_handlers:
            signal.signal(k, v)
        for s in self.signals_received:
            self.old_handlers[s[0]](*s)


def s3_download_key(filename, HCP_local=HCP_FOLDER, s3_bucket=s3_bucket):
    subject_base_filename = op.join(*filename.split('/')[1:])
    local_filename = op.join(HCP_local, subject_base_filename)
    if not op.exists(local_filename):
        print('Download', local_filename)
        k = s3_bucket.get_key(filename)
        if k is None:
            return False
        try:
            os.makedirs(
                op.dirname(local_filename),
                exist_ok=True, mode=0o0770
            )
            with open(local_filename, 'wb') as f:
                k.get_contents_to_file(f)
        except OSError as err:
            print(str(err))
            return str(err)
        return True
    else:
        return False


out = Parallel(n_jobs=N_JOBS, verbose=5)(
   delayed(s3_download_key)(key.name)
   for key in s3_bucket.list("HCP_1200")
   if (
       # fnmatch.fnmatch(key.name, '*/MNINonLinear/*T1w*restore_brain*nii.gz')
       # or fnmatch.fnmatch(key.name, '*/MNINonLinear/*parc.nii.gz')
       # or fnmatch.fnmatch(key.name, '*/MNINonLinear/*32k*surf.gii')
       # or fnmatch.fnmatch(key.name, '*/MNINonLinear/*sulc*32k*')
       # or fnmatch.fnmatch(key.name, '*/MNINonLinear/*parc*32k')
       fnmatch.fnmatch(key.name, '*/T1w/T1w_acpc_dc_restore_brain.nii.gz')
       or fnmatch.fnmatch(key.name, '*/T1w/aparc.a2009s+aseg.nii.gz')
       or fnmatch.fnmatch(key.name, '*/T1w/ribbon.nii.gz')
       # or fnmatch.fnmatch(key.name, '*/T1w/*sulc*32k*')
       # or fnmatch.fnmatch(key.name, '*/T1w/*parc*32k')
   )
)

missing_subjects = set([
    re.findall('[0-9][0-9][0-9]+', o)[0]
    for o in out if not isinstance(o, bool)
])

print("Missing subjects {}".format(len(missing_subjects)))
