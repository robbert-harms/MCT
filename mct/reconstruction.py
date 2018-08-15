import logging
import timeit
from contextlib import contextmanager
import numpy as np
import collections

import time

import mdt
import mot
from mct.__version__ import VERSION
from mct.utils import UnzippedNiftis


__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ReconstructionMethod(object):
    """A reconstruction method reconstructs volume(s) from multiple channels."""

    command_line_info = '''
        No info defined for this method.
    '''

    def reconstruct(self, output_directory, volumes=None):
        """Reconstruct the given channels and place the result in a subdirectory in the given directory.

        Args:
            output_directory (str): the location for the output files
            volumes (list of int): the indices of the volume we want to reconstruct (0-based).

        Returns:
             dict: the set of results from this reconstruction method
        """
        raise NotImplementedError()


class SliceBySliceReconstructionMethod(ReconstructionMethod):

    def __init__(self, channels, **kwargs):
        """Create a basic reconstruction method initialized with the given data and settings.

        Args:
            channels (list): the list of input nifti files, one for each channel element. Every nifti file
                    should be a 4d matrix with on the 4th dimension the time series. The length of this list
                    equals the number of channels.
            slicing_axis (int): the (x,y,z) axis over which we will loop to reconstruct the volumes. 0=x, 1=y, 2=z.
        """
        super().__init__()
        self._channels = channels
        self._logger = logging.getLogger(__name__)
        self._output_subdir = self.__class__.__name__
        self._slicing_axis = kwargs.get('slicing_axis', 2)

    def reconstruct(self, output_directory, volumes=None):
        output_subdir = output_directory + '/' + self._output_subdir
        niftis = UnzippedNiftis(self._channels, output_subdir)

        combined = self._reconstruct(niftis, volumes=volumes)

        if isinstance(combined, collections.Mapping):
            for name, data in combined.items():
                mdt.write_nifti(data, output_subdir + '/{}.nii.gz'.format(name), niftis[0].get_header())
        else:
            mdt.write_nifti(combined, output_subdir + '/reconstruction.nii.gz', niftis[0].get_header())

    def _reconstruct(self, input_niftis, volumes=None):
        nifti_shape = input_niftis[0].shape
        slice_results = []

        self._logger.info('Using MCT version {}'.format(VERSION))
        self._logger.info('Using MOT version {}'.format(mot.__version__))

        if volumes is None:
            volumes = list(range(nifti_shape[3]))
            self._logger.info('Reconstructing all volumes')
        elif volumes == 'odd':
            volumes = list(range(0, nifti_shape[3], 2))
            self._logger.info('Reconstructing all odd volumes {}'.format(list(volumes)))
        elif volumes == 'even':
            volumes = list(range(1, nifti_shape[3], 2))
            self._logger.info('Reconstructing all even volumes {}'.format(list(volumes)))
        else:
            self._logger.info('Reconstructing using the specified volumes {}'.format(list(volumes)))

        start_time = timeit.default_timer()
        logging_enabled = True
        for z_slice in range(nifti_shape[2]):
            self._logger.info(self._get_batch_start_message(start_time, z_slice, nifti_shape[2]))

            slice_data = self._get_slice_all_channels(input_niftis, z_slice, volumes)

            if logging_enabled:
                slice_results.append(self._reconstruct_slice(slice_data, z_slice))
                logging_enabled = False
            else:
                with self._with_logging_to_debug():
                    slice_results.append(self._reconstruct_slice(slice_data, z_slice))

        self._logger.info('Computed all slices, now assembling results.')

        if isinstance(slice_results[0], collections.Mapping):
            final_results = {}
            for key in slice_results[0]:
                final_results[key] = np.stack([el[key] for el in slice_results], axis=self._slicing_axis)
            return final_results

        return np.stack(slice_results, axis=self._slicing_axis)

    @contextmanager
    def _with_logging_to_debug(self):
        handlers = logging.getLogger('mot').handlers
        for handler in handlers:
            handler.setLevel(logging.WARNING)
        yield
        for handler in handlers:
            handler.setLevel(logging.INFO)

    def _get_batch_start_message(self, start_time, slice_index, total_nmr_slices):
        def calculate_run_days(runtime):
            if runtime > 24 * 60 * 60:
                return runtime // 24 * 60 * 60
            return 0

        run_time = timeit.default_timer() - start_time
        current_percentage = slice_index / float(total_nmr_slices)
        if current_percentage > 0:
            remaining_time = (run_time / current_percentage) - run_time
        else:
            remaining_time = None

        run_time_str = str(calculate_run_days(run_time)) + ':' + time.strftime('%H:%M:%S', time.gmtime(run_time))
        remaining_time_str = (str(calculate_run_days(remaining_time)) + ':' +
                              time.strftime('%H:%M:%S', time.gmtime(remaining_time))) if remaining_time else '?'

        return ('Processing slice {0} of {1}, we are at {2:.2%}, time spent: {3}, time left: {4} (d:h:m:s).'.
                format(slice_index, total_nmr_slices, current_percentage, run_time_str, remaining_time_str))

    def _reconstruct_slice(self, slice_data, slice_index):
        """Reconstruct the given slice.

        Args:
            slice_data (ndarray): a 4d array with the first two dimensions the remaining voxel locations and
                third the volumes and finally the channels
            slice_index (int): the slice index

        Returns:
            ndarray: the reconstruction of the given volumes for the given slice. This should be an array of
                shape (dim1, dim2, nmr_volumes) where dim1 and dim2 should be the same dimensions as
                the input (typically x and y if we are looping over z) and nmr_volumes equals the number of volumes.
        """
        raise NotImplementedError()

    def _get_slice_all_channels(self, input_niftis, slice_index, volumes):
        """Get the requested slice over each of the input niftis.

        Args:
            input_niftis (list of nifti): the list of nifti file objects
            slice_index (int): the slice index we want
            volumes (list of int): the volumes to use for the reconstruction

        Returns:
            ndarray: a 4d array with the first two dimensions the remaining voxel locations and then the timeseries
                and then the channels
        """
        if self._slicing_axis == 0:
            slices = [nifti.dataobj[int(slice_index), :, :][..., volumes] for nifti in input_niftis]
        elif self._slicing_axis == 1:
            slices = [nifti.dataobj[:, int(slice_index), :][..., volumes] for nifti in input_niftis]
        else:
            slices = [nifti.dataobj[:, :, int(slice_index)][..., volumes] for nifti in input_niftis]
        return np.stack(slices, axis=-1)
