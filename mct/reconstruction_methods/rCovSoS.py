from textwrap import dedent
import numpy as np
from numpy.linalg import inv, cholesky
from mct.reconstruction import SliceBySliceReconstructionMethod
from mct.utils import load_nifti

__author__ = 'Francisco Javier Fritz, Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class rCovSoS(SliceBySliceReconstructionMethod):

    command_line_info = dedent('''
        Root of the Covariance Sum Of Squares reconstruction [Triantafyllou 2016 and Pruesmann 2008].

        Required args:
            Covariance noise matrix (complex square matrix with dimension (N, N) with N equal to the number of channels)

        Optional keyword args:
            None
        ''')

    def __init__(self, channels, covariance_noise_matrix, **kwargs):
        """Instantiate the rCovSos method.

        This will do a cholesky decomposition on the input covariance noise matrix, inverts it and takes the
        transpose. We then take per voxel the dot product of the signals with this resulting matrix.

        Args:
            channels (list): the list of input nifti files, one for each channel element. Every nifti file
                    should be a 4d matrix with on the 4th dimension all the time series. The length of this list
                    should equal the number of input channels.
            covariance_noise_matrix (str or ndarray): the corresponding noise matrix to use. If a string is given it is
                supposed to be a nifti file path.
        """
        super().__init__(channels, **kwargs)
        if isinstance(covariance_noise_matrix, str):
            covariance_noise_matrix = load_nifti(covariance_noise_matrix).get_data()
        self._inverse_covar = cholesky(inv(covariance_noise_matrix))

    def _reconstruct_slice(self, slice_data, slice_index):
        return np.sqrt(np.sum(np.abs(np.dot(slice_data, self._inverse_covar)) ** 2, axis=-1))
