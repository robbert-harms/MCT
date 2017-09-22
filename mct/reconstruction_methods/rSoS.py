from textwrap import dedent
import numpy as np
from mct.reconstruction import SliceBySliceReconstructionMethod

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class rSoS(SliceBySliceReconstructionMethod):

    command_line_info = dedent('''
        Typical root Sum Of Squares reconstruction.

        Required args:
            None

        Optional keyword args:
            None
        ''')

    def _reconstruct_slice(self, slice_data, slice_index):
        return np.sqrt(np.sum(np.abs(slice_data).astype(np.float64) ** 2, axis=-1))
