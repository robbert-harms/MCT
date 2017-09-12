import os
from contextlib import contextmanager
import time
import nibabel as nib
import six

from mot.load_balance_strategies import EvenDistribution

import mot
import numpy as np
import mdt
import collections
from multiprocessing import Process
from mdt.nifti import nifti_filepath_resolution, unzip_nifti
from mdt.utils import split_image_path


__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class BufferedInputReader(Process):

    def __init__(self, input_niftis, input_queue, in_reading_queue, output_queue):
        """This processor reads in parts of the nifti files and buffers it into a queue.

        The idea is that while some of the other voxels are being optimized, we can already load in the next batch
        since IO may take some time. To do so, we use the multiprocessing library from Python and implement two queues,
        one (the input queue) for sending the indices of the voxels we want and the second (the output queue) for
        retrieving the image data of those indices.

        Args:
            input_niftis (list of nibabel niftis): the list of input nifti files
            input_queue (multiprocessing.queue): the input queue listening for input volume indices. Send None to
                this queue to stop the process.
            in_reading_queue (multiprocessing.queue): the queue used for the input reader to communicate it is
                reading the next batch. The reader will push a value to this queue to signal that it is
                in the process of reading the next batch. It will pop the value after it pushed the new batch to the
                output queue.
            output_queue (multiprocessing.queue): the queue to send the output batch data to.
        """
        super(BufferedInputReader, self).__init__()
        self._input_niftis = input_niftis

        self._input_queue = input_queue
        self._in_reading_queue = in_reading_queue
        self._output_queue = output_queue

        self._nmr_channels = len(self._input_niftis)
        self._nmr_volumes = self._input_niftis[0].shape[3]
        self._input_dtype = self._input_niftis[0].get_data_dtype()

    def run(self):
        while True:
            volume_indices = self._input_queue.get()

            with self._hold_in_reading_queue():
                if volume_indices is None:
                    return

                index_tuple = tuple(volume_indices[..., ind] for ind in range(3))
                batch = np.zeros((volume_indices.shape[0], self._nmr_volumes, self._nmr_channels),
                                 dtype=self._input_dtype)

                for channel_ind, nifti in enumerate(self._input_niftis):
                    batch[..., channel_ind] = nifti.get_data()[index_tuple]

                self._output_queue.put(batch)

                while self._output_queue.empty():
                    time.sleep(0.01)

    @contextmanager
    def _hold_in_reading_queue(self):
        self._in_reading_queue.put(True)
        yield
        self._in_reading_queue.get()


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

    The indices of the devices can be used in the model fitting/sampling functions for 'cl_device_ind'.

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
    return mot.configuration.RuntimeConfigurationAction(
        cl_environments=cl_envs,
        load_balancer=EvenDistribution())


def combine_weighted_sum(input_channels, weights, output_filename):
    """Combine all the coils using the given weights.

    Args:
        input_channels (list of str): the list with the input channel filenames
        weights (str or ndarray): the weights to use for the reconstruction.
        output_filename (str): the output filename
    """
    final_shape = load_nifti(input_channels[0]).shape
    output = np.zeros(final_shape, dtype=np.float64)

    if isinstance(weights, six.string_types):
        weights = load_nifti(weights).get_data()

    for ind, input_channel in enumerate(input_channels):
        output += weights[..., ind, None] * load_nifti(input_channel).get_data()

    header = load_nifti(input_channels[0]).get_header()
    mdt.write_nifti(output, header, output_filename)


def extract_timepoints(input_files, timepoints, output_dir):
    """Extract specific time points from all input nifti files.

    Args:
        input_files (list[str]): the list with input filenames
        timepoints (str or range): either a sequence of specific timepoints to extract or the string "even" for
            the even timepoints (starting at 0) or the string "odd" with the odd timepoints (starting at 1).
        output_dir (str): the location for the output files. Since the filenames will be exactly the same as the
            input filenames, we need a new directory for the output files.
    """
    output_dir = output_dir.replace('//', '/')
    nmr_timepoints = load_nifti(input_files[0]).shape[3]

    if timepoints == 'even':
        timepoints = range(0, nmr_timepoints, 2)
    elif timepoints == 'odd':
        timepoints = range(1, nmr_timepoints, 2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in input_files:
        full_name = nifti_filepath_resolution(filename)
        nifti = load_nifti(full_name)
        output_path = output_dir + '/' + ''.join(split_image_path(full_name)[1:])
        mdt.write_nifti(nifti.get_data()[..., timepoints], nifti.get_header(), output_path)


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    This will apply path resolution if a filename without extension is given. See the function
    :func:`nifti_filepath_resolution` for details.

    Args:
        nifti_volume (string): The filename of the volume to use.

    Returns:
        :class:`nibabel.nifti1.Nifti1Image`
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nib.load(path)
