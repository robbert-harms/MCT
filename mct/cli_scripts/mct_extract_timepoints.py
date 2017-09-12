#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Extract the provided timepoints from all the channels.

This is necessary if you want to use a specific subset of your acquired timeseries in the reconstruction. The idea is that you extract the timepoints first and then reconstruct normally on the extracted points.

Please note that this will overwrite any existing files in the output directory. Make sure you choose a different output directory than your input directory.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter

from mct import extract_timepoints
from mdt.nifti import nifti_filepath_resolution
from mdt.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExtractTimepoints(BasicShellApplication):

    def __init__(self):
        super(ExtractTimepoints, self).__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-extract-timepoints 0.nii 1.nii 2.nii -t 0 1 2 3 4 -o ./output
            mct-extract-timepoints channel_{0..16}.nii -t odd -o ./output
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_files', type=str, nargs='+',
                            help='the input channels')

        parser.add_argument('-t', '--timepoints', required=True, nargs='+',
                            help='the timepoints to extract, either a list of indices or the literal "odd" or "even"')

        parser.add_argument('-o', '--output_folder', required=True,
                            help='the directory for the output').completer = FilesCompleter()

        return parser

    def run(self, args, extra_args):
        input_files = get_input_files(args.input_files, os.path.realpath(''))

        output_folder = os.path.realpath(args.output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        timepoints = []
        if args.timepoints[0] == 'odd':
            timepoints = 'odd'
        elif args.timepoints[0] == 'even':
            timepoints = 'even'
        else:
            for timepoint in args.timepoints:
                timepoints.append(int(timepoint))

        extract_timepoints(input_files, timepoints, output_folder)


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
    return ExtractTimepoints().get_documentation_arg_parser()


if __name__ == '__main__':
    ExtractTimepoints().start()
