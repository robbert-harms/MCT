from textwrap import dedent

import numpy as np
from mct.processing import ReconstructionMethod
from mct.utils import calculate_tsnr

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class rSoS(ReconstructionMethod):

    command_line_info = dedent('''
        Typical root Sum Of Squares reconstruction.

        Required args:
            None

        Optional keyword args:
            None
        ''')

    def reconstruct(self, batch, volume_indices):
        reconstruction = np.sqrt(np.sum(np.abs(batch).astype(np.float64) ** 2, axis=2))
        return {'reconstruction': reconstruction, 'tSNR': calculate_tsnr(reconstruction)}
