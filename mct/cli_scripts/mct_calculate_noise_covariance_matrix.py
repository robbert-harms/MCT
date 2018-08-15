#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Obtain the noise covariance matrix from noise adjustment (noiseadj) data, 2D, 3D or 4D noise volumes obtained by imaging at 0 voltage.

This is necessary if you want to do rCovSos or Roemer reconstruction in your data. The covariance matrix is a complex matrix with square size n, where n is the number of channels of the coil
used to acquire your data. IT IS IMPORTANT that the order of the channels in your data IS the same order of the channels in the covariance noise matrix.

This will overwrite the output file if it exists.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter
import mdt

from mct import load_nifti
from mct.utils import calculate_noise_covariance_matrix
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Francisco.Lagos'


class EstimateNoiseCovMatrix(BasicShellApplication):

    def __init__(self):
        super().__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-calculate-noise-covariance-matrix input.nii -o ./noise_covar.nii
            mct-calculate-noise-covariance-matrix input.nii --normalize -o ./noise_covar.nii
        ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_file', type=str, nargs='+',
                            help='the input data which can be noiseadj (1D), or 0V noise volume (2D or 3D)')

        parser.add_argument('--normalize', dest='normalize', action='store_true',
                            help='normalize the covariance noise matrix, in which the diagonal '
                                 'elements are unitary value')
        parser.set_defaults(normalize=False)

        parser.add_argument('-o', '--output_file', required=True,
                            help='the output filename').completer = FilesCompleter()

        return parser

    def run(self, args, extra_args):
        input_file = os.path.realpath(args.input_file)
        output_file = os.path.realpath(args.output_file)

        input_data = load_nifti(input_file).get_data()
        header = load_nifti(input_file).get_header()

        noise_covar = calculate_noise_covariance_matrix(input_data, normalize=args.normalize)

        mdt.write_nifti(noise_covar, header, output_file)


def get_doc_arg_parser():
    return EstimateNoiseCovMatrix().get_documentation_arg_parser()

if __name__ == '__main__':
    EstimateNoiseCovMatrix().start()
