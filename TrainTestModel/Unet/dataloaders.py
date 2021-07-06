import os
import torch
import torchio
from TrainTestModel.Unet.utils import get_logger

manual_seed = None

logger = get_logger(__name__)

def _get_dataset(loaders_config, train_or_test):

    ############## Crear Sujetos ###############

    if train_or_test == 'train':
        # lista de los directorios donde estarán todas las imágenes
        assert 'train_path' in loaders_config, 'No se especificó directorio de las imágenes de entrenamiento'
        path = loaders_config.get('train_path')
    elif train_or_test == 'val':
        assert 'val_path' in loaders_config, 'No se especificó directorio de las imágenes de validación'
        path = loaders_config.get('val_path')
    else:
        assert 'test_path' in loaders_config, 'No se especificó directorio de las imágenes de test'
        path = loaders_config.get('test_path')

    reg = loaders_config.get('reg')

    dirlist = []
    subj_id = []
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)):
            subj_id.append(filename)
            dirlist.append( os.path.join(path, filename, reg) )

    dirlist.sort()
    subj_id.sort()

    name_in_image = loaders_config.get('in_image')
    name_gt_image = loaders_config.get('gt_image')
    data_augment = loaders_config.get('data_augment')
    logger.info(f'Name of input image: {name_in_image}')
    logger.info(f'Name of ground truth image: {name_gt_image}')
    logger.info(f'Data augmentation: {data_augment}')

    img_paths, lbl_paths = zip(*[ (os.path.join(dir, name_in_image), os.path.join(dir, name_gt_image)) for dir in dirlist ])

    assert len(img_paths) == len(lbl_paths), "la cantidad de imagenes debe ser igual al de etiquetas"

    subjects = []
    for (img_path, lbl_path, id) in zip(img_paths, lbl_paths, subj_id):
        subject_dict = {
            'image': torchio.Image(img_path, torchio.INTENSITY),
            'label': torchio.Image(lbl_path, torchio.LABEL),
            'subj_id' : id,
            'img_path' : img_path,
            'lbl_path' : lbl_path,
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)

    # data augmentation para los datos de entrenamiento
    transform = None
    if train_or_test == 'train' and data_augment:
        rot = torchio.RandomAffine(degrees=10, center='image', p=0.3, default_pad_value='otsu', seed=manual_seed)
        transform = torchio.Compose(
            # transforms=[flip, rot],
            transforms=[rot],
            p=1
        )

    dataset = torchio.SubjectsDataset(subjects, transform=transform)

    return dataset


def get_patched_loaders(loaders_config, tr_set, val_set):
    ############## Entrenamiento basado en Parches ##############

    patch_size = loaders_config.get('patch_size')
    max_queue_length = loaders_config.get('max_queue_length')
    samples_per_volume = loaders_config.get('samples_per_volume')

    # labels = loaders_config.get('labels')
    label_probabilities = loaders_config.get('label_probabilities')

    #  A sampler used to extract patches from the volumes.
    sampler = torchio.sampler.LabelSampler(patch_size,
                                           label_name='label',
                                           label_probabilities=label_probabilities)
    # label_probabilities, doble de probabilidad de que el parche esté centrado en la etiqueta 1 que en la 0

    logger.info(f'Patch size: {patch_size}')
    logger.info(f'Sampler: {sampler.__class__.__name__}')
    logger.info(f'samples_per_volume: {samples_per_volume}')
    logger.info(f'max_queue_length: {max_queue_length}')

    patches_tr_set = torchio.Queue(
        subjects_dataset=tr_set,
        max_length=max_queue_length,  # Maximum number of patches that can be stored in the queue.
        samples_per_volume=samples_per_volume,  # Number of patches to extract from each subject.
        sampler=sampler,
        # num_workers=n_workers, # aca no funciona, mejor usarlo en Dataloader
        shuffle_subjects=False,  # no afecta tiempo de entrenamiento
        shuffle_patches=False,  # no afecta tiempo de entrenamiento
    )

    patches_val_set = torchio.Queue(
        subjects_dataset=val_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        # num_workers=n_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    return patches_tr_set, patches_val_set



def get_train_loaders(config, seed):
    global manual_seed
    manual_seed = seed

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')
    tr_set = _get_dataset(loaders_config, 'train')
    val_set = _get_dataset(loaders_config, 'val')

    whole_image = loaders_config.get('whole_image')
    batch_size = loaders_config.get('batch_size', 1)
    n_workers = loaders_config.get('n_workers', 1)
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Number of workers for the dataloader: {n_workers}')

    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()
        config['batch_size'] = batch_size

    # Entrenar con parches
    if not whole_image:
        logger.info(f'Training with patches')
        # Crear el conjunto de entrenamiento y validacion con parches
        patched_tr_set, patched_val_set = get_patched_loaders(loaders_config, tr_set, val_set)
        tr_set = patched_tr_set
        val_set = patched_val_set
    # Entrenar con la imagen completa
    else:
        logger.info(f'Training with whole image')

    # Crear el dataloader
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    return {
        'train': tr_loader,
        'val': val_loader
    }


def get_test_subj_or_loader(loaders_config):

    logger.info('Creating test subject set...')

    return _get_dataset(loaders_config, 'test')










