#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Split the volumes on the given axis.

Since the reconstruction method requires you to have one nifti file per channel, you need to split it if you have all your channels in one volume.

The axis are indexed zero-based. To use the last dimension use -1. By default it will split on the last dimension.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter
from mct.utils import split_write_volumes
from mdt.lib.nifti import nifti_filepath_resolution
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SplitVolumes(BasicShellApplication):

    def __init__(self):
        super().__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-split-volumes volume.nii --axis=-1
            mct-split-volumes volume.nii --axis=3
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_file', type=str, help='the input channels')
        parser.add_argument('-a', '--axis', default=-1, help='the axis to use for the split.')
        parser.add_argument('-o', '--output_folder', default=None,
                            help='the output base directory, the items '
                                 'will be split numerically.').completer = FilesCompleter()

        return parser

    def run(self, args, extra_args):
        input_file = os.path.realpath(get_input_file(args.input_file, os.path.realpath('')))
        dirname, _, _ = split_image_path(input_file)

        if args.output_folder is None:
            output_folder = dirname + '/split'
        else:
            output_folder = os.path.realpath(args.output_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        split_write_volumes(input_file, output_folder, args.axis)


def get_input_file(input_file, base_dir):
    try:
        return nifti_filepath_resolution(input_file)
    except ValueError:
        return nifti_filepath_resolution(base_dir + '/' + input_file)


def get_doc_arg_parser():
    return SplitVolumes().get_documentation_arg_parser()


if __name__ == '__main__':
    SplitVolumes().start()
