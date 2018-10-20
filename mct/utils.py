import os
import nibabel as nib

import mot
import numpy as np
import mdt
import collections
from mdt.lib.nifti import nifti_filepath_resolution, unzip_nifti
from mdt.utils import split_image_path


__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class UnzippedNiftis(collections.Sequence):

    def __init__(self, input_filenames, tmp_dir):
        """Given a list of nifti filenames, this sequence will expose them as (unzipped) nibabel nifti files.

        That is, each element of this sequence is a loaded nifti file, loaded using ``load_nifti(file_name)``.
        If one or more of the input niftis are zipped, this class will unzip them to the temporary location
        and then exposes those as the opened nifti files. At deconstruction, the unzipped niftis will be removed.

        Args:
            input_filenames (list of str): the list of input filenames
            tmp_dir (str): the location for storing unzipped versions of zipped nifti files
        """
        self._input_filenames = input_filenames
        self._tmp_dir = tmp_dir
        self._files_to_remove = []
        self._niftis = self._load_niftis()

    def _load_niftis(self):
        niftis = []
        for filename in self._input_filenames:
            filename = nifti_filepath_resolution(filename)
            _, basename, extension = split_image_path(filename)

            if extension.endswith('gz'):
                unzipped_nifti_path = os.path.join(self._tmp_dir, basename + '.nii')
                unzip_nifti(filename, unzipped_nifti_path)

                self._files_to_remove.append(unzipped_nifti_path)
                resolved_path = unzipped_nifti_path
            else:
                resolved_path = filename

            niftis.append(load_nifti(resolved_path))
        return niftis

    def __del__(self):
        for filename in self._files_to_remove:
            if os.path.isfile(filename):
                os.remove(filename)

    def __len__(self):
        return len(self._input_filenames)

    def __getitem__(self, index):
        return self._niftis[index]


def calculate_tsnr(data, axis=-1):
    """Calculate the tSNR of the given data.

    By default this will calculate the tSNR on the last axis of the input array. The tSNR is defined as the
    ``mean(data) / std(data)``.

    Args:
        data (ndarray): the data we want to calculate the tSNR off
        axis (int): the axis on which to compute the tSNR
    """
    return np.mean(data, axis=axis) / np.std(data, axis=axis)


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting/sample functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    from mdt.utils import get_cl_devices
    return get_cl_devices()


def get_mot_config_context(cl_device_ind):
    """Get the configuration context that uses the given devices by index.

    Args:
        cl_device_ind (int or list of int): the device index or a list of device indices

    Returns:
        mot.configuration.ConfigAction: the configuration action to use
    """
    if cl_device_ind is not None and not isinstance(cl_device_ind, collections.Iterable):
        cl_device_ind = [cl_device_ind]

    if cl_device_ind is None:
        return mot.configuration.VoidConfigurationAction()

    cl_envs = [get_cl_devices()[ind] for ind in cl_device_ind]
    return mot.configuration.RuntimeConfigurationAction(cl_environments=cl_envs)


def combine_weighted_sum(input_channels, weights, output_filename):
    """Combine all the coils using the given weights.

    Args:
        input_channels (list of str): the list with the input channel filenames
        weights (str or ndarray): the weights to use for the reconstruction.
        output_filename (str): the output filename
    """
    final_shape = load_nifti(input_channels[0]).shape
    output = np.zeros(final_shape, dtype=np.float64)

    if isinstance(weights, str):
        weights = load_nifti(weights).get_data()

    for ind, input_channel in enumerate(input_channels):
        output += weights[..., ind, None] * load_nifti(input_channel).get_data()

    header = load_nifti(input_channels[0]).get_header()
    mdt.write_nifti(output, output_filename, header)


def calculate_noise_covariance_matrix(input_data, normalize=False):
    """Obtain noise covariance matrix from raw data.

    The input data (the raw data) is expected to be in complex k-space or x-space.

    This function supports 1d, 2d, 3d noise volumes as input and assumes that the noise data of the channels are in the
    last dimension.

    Args:
        input_data (str): the input data
        normalise (bool): If True, then the calculated noise matrix will be normalised.

    Returns:
        ndarray: a square matrix of the number of channels
    """
    nmr_elements_per_channel = np.prod(input_data.shape[:-1])

    data = np.reshape(input_data, (-1, input_data.shape[-1]))
    noise_matrix = np.dot(np.conj(data.T), data)/(nmr_elements_per_channel * input_data.shape[-1])

    if normalize:
        scaled_data = data / np.sqrt(np.diag(noise_matrix))
        noise_matrix = np.dot(np.conj(scaled_data.T), scaled_data)/(nmr_elements_per_channel * input_data.shape[-1])

    return noise_matrix


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    This will apply path resolution if a filename without extension is given. See the function
    :func:`nifti_filepath_resolution` for details.

    Args:
        nifti_volume (string): The filename of the volume to use.

    Returns:
        :class:`nibabel.spatialimages.SpatialImage`
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nib.load(path)


def split_write_volumes(input_file, output_dir, axis=-1):
    """Split and write the given input volume to separate nifti files.

    Args:
        input_file (str): the input nifti file to split
        output_dir (str): the output directory, this will write the split volumes to that directory
            with the same basename as the input file, with the slice index appended
        axis (int): the axis to split on
    """
    dirname, basename, extension = split_image_path(input_file)

    nifti = load_nifti(input_file)
    data = nifti.get_data()
    header = nifti.get_header()

    for ind in range(data.shape[axis]):
        index = [slice(None)] * len(data.shape)
        index[axis] = ind
        mdt.write_nifti(data[index], '{}/{}_{}.nii'.format(output_dir, basename, ind), header)

