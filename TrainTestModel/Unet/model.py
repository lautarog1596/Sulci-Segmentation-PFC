from unet import UNet
# from Unet.unet import UNet
from TrainTestModel.Unet.utils import get_logger

logger = get_logger(__name__)

def get_model(config):

    model_config = config['model']

    in_channels = model_config.get('in_channels')
    out_classes = model_config.get('out_classes')
    dimensions = model_config.get('dimensions')
    num_encoding_blocks = model_config.get('num_encoding_blocks')
    out_channels_first_layer = model_config.get('out_channels_first_layer')
    normalization = model_config.get('normalization')
    # kernel_size = model_config.get('kernel_size')
    pooling_type = model_config.get('pooling_type')
    upsampling_type = model_config.get('upsampling_type')
    preactivation = model_config.get('preactivation')
    residual = model_config.get('residual')
    padding = model_config.get('padding')
    padding_mode = model_config.get('padding_mode')
    activation = model_config.get('activation')
    initial_dilation = model_config.get('initial_dilation', None)
    dropout = model_config.get('dropout')
    monte_carlo_dropout = model_config.get('monte_carlo_dropout')

    model = UNet(
        in_channels=in_channels,
        out_classes=out_classes,  # clases binarias
        dimensions=dimensions,
        num_encoding_blocks=num_encoding_blocks,
        out_channels_first_layer=out_channels_first_layer,  # 64
        normalization= normalization,
        pooling_type= pooling_type,
        upsampling_type=upsampling_type,
        preactivation= preactivation,
        residual= residual,
        padding= padding,
        padding_mode= padding_mode,
        activation= activation,
        initial_dilation= initial_dilation,
        dropout= dropout,
        monte_carlo_dropout= monte_carlo_dropout
    )

    logger.info(model)

    return model
