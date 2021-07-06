import random
import shutil, os
import pickle
import sys


# ruta en donde están guardadas las carpetas con las imagenes y se almacenaran las de entrenamiento
train_path = '/mnt/datos_facultad/datos_sinc_local/HCP_1200/Train/'
# ruta en donde guardar las de validacion
val_path = '/mnt/datos_facultad/datos_sinc_local/HCP_1200/Val/'
# ruta en donde guardar las de test
test_path = '/mnt/datos_facultad/datos_sinc_local/HCP_1200/Test/'


if not os.path.isdir(train_path):
    msg = 'no existe el directorio ' + train_path + ' donde mover las imágenes de entrenamiento'
    sys.exit(msg)
if not os.path.isdir(val_path):
    msg = 'no existe el directorio ' + val_path + ' donde mover las imágenes de validacion'
    sys.exit(msg)
if not os.path.isdir(test_path):
    msg = 'no existe el directorio ' + test_path + ' donde mover las imágenes de test'
    sys.exit(msg)


# lista de las carpetas con las imágenes
dirlist = [os.path.join(train_path, filename) for filename in os.listdir(train_path)
           if (os.path.isdir(os.path.join(train_path, filename)))]
dirlist.sort()

# guardar la lista en "all_data.txt"
try:
    dirlist_loaded = pickle.load(open("all_data.txt", "rb"))
    if dirlist_loaded != dirlist:
        msg = 'las imágenes que tiene guardadas en: ' + train_path + ', son distintas a las ya guardades en: all_data.txt'
        sys.exit(msg)
except (OSError, IOError) as e:
    pickle.dump(dirlist, open("all_data.txt", "wb"))


# mezclar las carpetas para crear las particiones: 70 train - 15 val - 15 test
random.shuffle(dirlist)
dirlist_val = dirlist[:165]
dirlist_test = dirlist[165:330]
dirlist_train = dirlist[330:]


# guardar las listas dirlist_train, dirlist_val y dirlist_test en los archivos train_ds.txt val_ds.txt test_ds.txt
try:
    train_ds_loaded = pickle.load(open("train_ds.txt", "rb"))
    val_ds_loaded = pickle.load(open("val_ds.txt", "rb"))
    test_ds_loaded = pickle.load(open("test_ds.txt", "rb"))
    if train_ds_loaded != dirlist_train or val_ds_loaded != dirlist_val or test_ds_loaded != dirlist_test:
        sys.exit('Ya hay particiones hechas distintas a las que quiere hacer')
except (OSError, IOError) as e:
    pickle.dump(dirlist_train, open("train_ds.txt", "wb"))
    pickle.dump(dirlist_val, open("val_ds.txt", "wb"))
    pickle.dump(dirlist_test, open("test_ds.txt", "wb"))
    for val in dirlist_val:
        shutil.move(val, val_path)
    for test in dirlist_test:
        shutil.move(test, test_path)
    sys.exit('Particiones creadas')

print('Ya hay particiones hechas iguales a las que intenta hacer')

