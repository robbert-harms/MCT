import logging
from contextlib import contextmanager
import numpy as np
import collections
import mdt
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

    @property
    def name(self):
        """The name of this method for use in output files."""
        raise NotImplementedError()

    def reconstruct(self, channels, output_directory):
        """Reconstruct the given channels according and place the result in a subdirectory in the given directory.

        Args:
            channels (list): the list of input nifti files, one for each channel element. Every nifti file
                should be a 4d matrix with on the 4th dimension all the time series. The length of this list
                should equal the number of input channels.
            output_directory (str): the location for the output files

        Returns:
             dict: the set of results from this reconstruction method
        """
        raise NotImplementedError()


class AbstractReconstructionMethod(ReconstructionMethod):

    def __init__(self, cl_environments=None):
        self._cl_environments = cl_environments
        self._logger = logging.getLogger(__name__)
        self._output_subdir = self.__class__.__name__

    def reconstruct(self, channels, output_directory, recalculate=False):
        output_subdir = output_directory + '/' + self._output_subdir
        niftis = UnzippedNiftis(channels, output_subdir)
        combined = self._reconstruct(niftis, output_subdir)

        if isinstance(combined, collections.Mapping):
            for name, data in combined.items():
                mdt.write_nifti(data, output_subdir + '/{}.nii.gz'.format(name), niftis[0].get_header())
        else:
            mdt.write_nifti(combined, output_subdir + '/reconstruction.nii.gz', niftis[0].get_header())

    def _reconstruct(self, input_niftis, output_directory):
        """To be overwritten by the implementing class."""
        raise NotImplementedError()


class SliceBySliceReconstructionMethod(AbstractReconstructionMethod):

    def __init__(self, cl_environments=None):
        super(SliceBySliceReconstructionMethod, self).__init__(cl_environments=cl_environments)
        self._slicing_axis = 2

    def _reconstruct(self, input_niftis, output_directory):
        nifti_shape = input_niftis[0].shape
        slice_results = []

        logging_enabled = True
        for z_slice in range(nifti_shape[2]):
            self._logger.info('Processing slice {} of {}'.format(z_slice, nifti_shape[2]))

            slice_data = self._get_slice_all_channels(input_niftis, z_slice)

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

    def _reconstruct_slice(self, slice_data, slice_index):
        """Reconstruct the given slice.

        Args:
            slice_data (ndarray): a 4d array with the first two dimensions the remaining voxel locations and
                third the timeseries and finally the channels
            slice_index (int): the slice index
        """
        raise NotImplementedError()

    def _get_slice_all_channels(self, input_niftis, slice_index):
        """Get the requested slice over each of the input niftis.

        Args:
            input_niftis (list of nifti): the list of nifti file objects
            slice_index (int): the slice index we want
            axis (Optional[Int]): the axis over which to loop

        Returns:
            ndarray: a 4d array with the first two dimensions the remaining voxel locations and then the timeseries
                and then the channels
        """
        if self._slicing_axis == 0:
            slices = [nifti.dataobj[int(slice_index), :, :] for nifti in input_niftis]
        elif self._slicing_axis == 1:
            slices = [nifti.dataobj[:, int(slice_index), :] for nifti in input_niftis]
        else:
            slices = [nifti.dataobj[:, :, int(slice_index)] for nifti in input_niftis]
        return np.stack(slices, axis=-1)
