#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Combine all the coil given some weights.

This will reconstruct the channels by summing the separate channels multiplied by the given weights.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter

from mct import combine_weighted_sum
from mdt.lib.nifti import nifti_filepath_resolution
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CombineWeightedSum(BasicShellApplication):

    def __init__(self):
        super().__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-comnbine-weighted-sum 0.nii 1.nii 2.nii -w weights.nii -o combined.nii
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_files', type=str, nargs='+',
                            help='the input channels')

        parser.add_argument('-w', '--weights', required=True,
                            help='the volume containing the weights (on the last dimension) to use '
                                 'for the reconstruction. ')

        parser.add_argument('-o', '--output_file', required=True,
                            help='the output filename').completer = FilesCompleter()

        return parser

    def run(self, args, extra_args):
        input_files = get_input_files(args.input_files, os.path.realpath(''))
        weights = get_input_files([args.weights], os.path.realpath(''))[0]

        output_file = os.path.realpath(args.output_file)
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        combine_weighted_sum(input_files, weights, output_file)


def get_input_files(input_files_listing, base_dir):
    input_files = []

    if input_files_listing:
        for filename in input_files_listing:
            try:
                input_files.append(nifti_filepath_resolution(filename))
            except ValueError:
                input_files.append(nifti_filepath_resolution(base_dir + '/' + filename))

    return input_files


def get_doc_arg_parser():
    return CombineWeightedSum().get_documentation_arg_parser()


if __name__ == '__main__':
    CombineWeightedSum().start()
