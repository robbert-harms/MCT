import logging
from multiprocessing import Queue

from mct.utils import BufferedInputReader, UnzippedNiftis
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

        self._input_queue = Queue()
        self._in_reading_queue = Queue()
        self._output_queue = Queue()
        self._input_reader = BufferedInputReader(self._input_niftis, self._input_queue,
                                                 self._in_reading_queue, self._output_queue)
        self._input_reader.start()
        self._logger = logging.getLogger('coil_combine')

    def __del__(self):
        if hasattr(self, '_input_reader'):
            self._input_reader.terminate()

    def finalize(self):
        self._input_queue.put(None)
        self._input_reader.join()
        super(ReconstructionProcessor, self).finalize()

    def combine(self):
        super(ReconstructionProcessor, self).combine()
        self._combine_volumes(self._output_dir, self._tmp_storage_dir, self._nifti_header)
        return create_roi(get_all_image_data(self._output_dir), self._mask)

    def _process(self, roi_indices, next_indices=None):
        volume_indices = self._volume_indices[roi_indices, :]

        next_volume_indices = None
        if next_indices is not None:
            next_volume_indices = self._volume_indices[next_indices]

        if self._output_queue.empty() and self._input_queue.empty() and self._in_reading_queue.empty():
            self._logger.info('Loading first batch.')
            self._input_queue.put(volume_indices)


        import time
        time.sleep(1)


        batch = self._output_queue.get()
        self._input_queue.put(next_volume_indices)

        if batch.size > 0:
            results = self._reconstruction_method.reconstruct(batch, volume_indices)
            self._write_volumes(results, roi_indices, self._tmp_storage_dir)
