import logging
from multiprocessing.pool import Pool

import numpy as np
from mct.utils import UnzippedNiftis
from mdt.nifti import get_all_image_data
from mdt.processing_strategies import SimpleModelProcessor
from mdt.utils import create_roi


__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ReconstructionMethod(object):
    """The reconstruction method encodes how to reconstruct the multiple channels.

    This works together with the :class:`~mct.reconstruction.ReconstructionProcessor` to reconstruct your images.
    The processing worker encodes how to load the input, the reconstruction method encodes how to reconstruct.
    """

    @property
    def name(self):
        """The name of this method for use in output files."""
        raise NotImplementedError()

    def reconstruct(self, batch, volume_indices):
        """Reconstruct the given batch which contains voxels at the specified indices.

        Args:
            batch (ndarray): a 3d matrix with (v, t, c) for voxels, timeseries, channels.
            volume_indices (ndarray): a 2d matrix with for every voxel in the batch the 3d location of that
                voxel in the original dimensions.

        Returns:
             dict: the set of results from this reconstruction method
        """
        raise NotImplementedError()


class ReconstructionProcessor(SimpleModelProcessor):

    def __init__(self, reconstruction_method, input_filenames, output_dir, mask):
        """Creates a processing routine for the coil combine reconstruction.

        This class takes care of loading the multiple channels data (in batches) and applying the given model on that
        for reconstruction.

        Args:
            reconstruction_method (ReconstructionMethod): the method to use for reconstructing the data
            input_filenames (list): the list of input nifti files, one for each channel. Every nifti file
                is a 4d matrix with on the 4th dimension the time series for that channel
            output_dir (str): the output location
            mask (str): a 3d masking matrix with the voxels to use in the reconstruction set on True
        """
        self._input_niftis = UnzippedNiftis(input_filenames, output_dir)

        super(ReconstructionProcessor, self).__init__(mask, self._input_niftis[0].get_header(),
                                                      output_dir, output_dir + '/tmp', True)

        self._reconstruction_method = reconstruction_method
        self._batch_loader = BatchLoader(self._input_niftis)
        self._logger = logging.getLogger('coil_combine')

    def combine(self):
        super(ReconstructionProcessor, self).combine()
        self._combine_volumes(self._output_dir, self._tmp_storage_dir, self._nifti_header)
        return create_roi(get_all_image_data(self._output_dir), self._mask)

    def _process(self, roi_indices, next_indices=None):
        volume_indices = self._volume_indices[roi_indices, :]
        batch = self._batch_loader.load_batch(volume_indices)
        if batch.size > 0:
            results = self._reconstruction_method.reconstruct(batch, volume_indices)
            self._write_volumes(results, roi_indices, self._tmp_storage_dir)


class BatchLoader(object):

    def __init__(self, input_niftis):
        """Helper class for loading the batches.

        Since this batch loader makes use of memory mapping, for optimal performance, make sure that the input
        niftis are unzipped.

        Args:
            input_niftis (list of nibabel niftis): the list of input nifti files
        """
        self._input_niftis = input_niftis
        self._nmr_channels = len(self._input_niftis)
        if len(self._input_niftis[0].shape) < 4:
            self._nmr_volumes = 1
        else:
            self._nmr_volumes = self._input_niftis[0].shape[3]
        self._input_dtype = self._input_niftis[0].get_data_dtype()

    def load_batch(self, volume_indices):
        """Load the batch at the given indices.

        Args:
            volume_indices (ndarray): a volume with the indices of the next voxels to load

        Returns:
            ndarray: a array of shape (v, t, c) with v the number of voxels, t the number of time serie volumes
                and c the number of channels.
        """
        # index_tuple = tuple(volume_indices[..., ind] for ind in range(3))
        # batch = np.zeros((volume_indices.shape[0], self._nmr_volumes, self._nmr_channels),
        #                  dtype=self._input_dtype)

        # for channel_ind, nifti in enumerate(self._input_niftis):
        #     data = nifti.dataobj
        #
        #     for lin_ind, slice_ind in enumerate(np.unique(index_tuple[0])):
        #         index_tuple_2d = volume_indices[volume_indices[:, 0] == slice_ind][:, 1:]
        #         data_slice = data[int(slice_ind), :, :, :]
        #         print(lin_ind, slice_ind)

        with Pool() as pool:
            result = pool.starmap(read_from_nifti,
                                  [(nifti.dataobj, volume_indices, self._nmr_volumes, self._input_dtype)
                                   for channel_ind, nifti in enumerate(self._input_niftis)])
        return np.stack(result, axis=-1)
        # for channel_ind, nifti in enumerate(self._input_niftis):
        #     batch[..., channel_ind] = read_from_nifti(nifti.dataobj, volume_indices,
        #                                               self._nmr_volumes, self._input_dtype)

            # data = nifti.dataobj
            # for lin_ind, vol_ind in enumerate(volume_indices):
            #     batch[lin_ind, :, channel_ind] = data[int(vol_ind[0]), int(vol_ind[1]), int(vol_ind[2])]


        # exit(0)

            # if len(data.shape) == 1:
            #     data = data[..., None]
            # batch[..., channel_ind] = data
            #
        # for channel_ind, nifti in enumerate(self._input_niftis):
        #     data = nifti.get_data()[index_tuple]
        #     if len(data.shape) == 1:
        #         data = data[..., None]
        #     batch[..., channel_ind] = data

        # return batch


def read_from_nifti(data, volume_indices, nmr_volumes, dtype):
    """Small helping routine """
    batch = np.zeros((volume_indices.shape[0], nmr_volumes), dtype=dtype)
    for lin_ind, vol_ind in enumerate(volume_indices):
        batch[lin_ind, :] = data[int(vol_ind[0]), int(vol_ind[1]), int(vol_ind[2])]
    return batch
