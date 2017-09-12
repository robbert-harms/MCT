import logging.config as logging_config
import os
import mot
import six
import yaml

import mdt
from mct.components_loader import load_reconstruction_method
from mct.processing import ReconstructionProcessor
from mct.utils import get_mot_config_context
from mdt.processing_strategies import VoxelRange
from .__version__ import VERSION, VERSION_STATUS, __version__
import numpy as np

from mct.utils import extract_timepoints, load_nifti, combine_weighted_sum
from mct.components_loader import load_reconstruction_method, get_reconstruction_method_class


__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"


try:
    config = '''
        version: 1
        disable_existing_loggers: False

        formatters:
            simple:
                format: "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s"

        handlers:
            console:
                class: mdt.log_handlers.StdOutHandler
                level: INFO
                formatter: simple

            dispatch_handler:
                class: mdt.log_handlers.LogDispatchHandler
                level: INFO
                formatter: simple

        loggers:
            coil_combine:
                level: DEBUG
                handlers: [console]

        root:
            level: INFO
            handlers: [dispatch_handler]
    '''
    logging_config.dictConfig(yaml.safe_load(config))

except ValueError:
    print('Logging disabled')


def reconstruct(reconstruction_method, input_filenames, output_dir, mask=None, max_batch_size=50000,
                cl_device_ind=None):
    """Reconstruct the MRI volumes using the desired reconstruction method.

    Args:
        reconstruction_method (str or ReconstructionMethod): the method to use for the reconstruction,
            if a string is given we try to look up the corresponding method by class name.
        input_filenames (list): the list of input nifti files, one for each channel element. Every nifti file
            is a 4d matrix with on the 4th dimension the time series for that channel
        output_dir (str): the output location
        mask (str): a mask or the filename of a mask. This expects voxels to process to be 1 or True and everything else
            to be False or 0 .
        max_batch_size (int): the maximum size per batch. Lower this to decrease memory usage. Increasing it might
            increase performance.
        cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
            get_cl_devices(). This can also be a list of device indices.
    """
    if isinstance(reconstruction_method, six.string_types):
        reconstruction_method = load_reconstruction_method(reconstruction_method)

    if mask is None:
        mask = np.ones(load_nifti(input_filenames[0]).shape[:3])
    elif isinstance(mask, six.string_types):
        mask = mdt.load_brain_mask(mask)

    data_shape = load_nifti(input_filenames[0]).shape
    if mask.shape != data_shape[:3]:
        raise ValueError('The shape of the mask "{}" does not equal the shape of the volumes "{}".'.format(
            mask.shape, data_shape[:3]))

    with mot.configuration.config_context(get_mot_config_context(cl_device_ind)):
        processor = ReconstructionProcessor(reconstruction_method, input_filenames,
                                            os.path.join(output_dir, reconstruction_method.__class__.__name__),  mask)
        processing_strategy = VoxelRange(max_nmr_voxels=max_batch_size)
        return processing_strategy.process(processor)
