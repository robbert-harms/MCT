from textwrap import dedent
import numpy as np
import numpy.linalg as linalg
import six

from mct.processing import ReconstructionMethod
from mct.utils import load_nifti

__author__ = 'Francisco Javier Fritz, Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class rCovSoS(ReconstructionMethod):

    command_line_info = dedent('''
        Root of the Covariance Sum Of Squares reconstruction [Tryantafullyi or whatever 2016].

        Required args:
            Covariance noise matrix (complex square matrix with dimension (N, N) with N equal to the number of channels)

        Optional keyword args:
            None
        ''')

    def __init__(self, covariance_noise_matrix):
        """Instantiate the rCovSos method.

        This will do a cholesky decomposition on the input covariance noise matrix, inverts it and takes the
        transpose. We then take per voxel the dot product of the signals with this resulting matrix.

        Args:
            covariance_noise_matrix (str or ndarray): the noise matrix to use. If a string is given it is
                supposed to be a nifti file path.
        """
        if isinstance(covariance_noise_matrix, six.string_types):
            covariance_noise_matrix = load_nifti(covariance_noise_matrix).get_data()
        else:
            covariance_noise_matrix = covariance_noise_matrix
        self._inverse_covar_tr = np.transpose(linalg.inv(linalg.cholesky(covariance_noise_matrix)))

    def reconstruct(self, batch, volume_indices):
        output = np.zeros(batch.shape[:2])

        for ind in range(batch.shape[1]):
            voxels = batch[:, ind, :]
            voxels = np.conj(voxels).T
            batch_noise_weighted = np.abs(np.conj(np.dot(self._inverse_covar_tr, voxels)).T)
            
            output[:, ind] = np.sqrt(np.sum(batch_noise_weighted ** 2, axis=1))

        return {'reconstruction': output}
