#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Calculate the tSNR of your timeseries.

By default this will calculate the tSNR on the last axis of the input array.

The tSNR is defined as the ``mean(data) / std(data)``.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter

import mdt
from mct.utils import calculate_tsnr
from mdt.utils import split_image_path
from mot.lib import cl_environments

import mct
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateTNSR(BasicShellApplication):

    def __init__(self):
        super().__init__()
        self.available_devices = list((ind for ind, env in
                                       enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection())))

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-calculate-tsnr data.nii
            mct-calculate-tsnr data.nii -o tSNR.nii
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_file', default=None,
                            help='the input file with the timeseries '
                                 'on the last dimension').completer = FilesCompleter()

        parser.add_argument('-o', '--output-file', default=None,
                            help='the output file for the tSNR data, defaults to a file in the directory'
                                 'of the input data').completer = FilesCompleter()
        return parser

    def run(self, args, extra_args):
        input_file = os.path.realpath(args.input_file)

        if args.output_file is None:
            dirname, basename, ext = split_image_path(input_file)
            output_file = dirname + basename + '_tSNR' + ext
        else:
            output_file = os.path.realpath(args.output_file)

        nifti = mct.load_nifti(input_file)
        tsnr = calculate_tsnr(nifti.get_data())
        mdt.write_nifti(tsnr, output_file, nifti.get_header())


def get_doc_arg_parser():
    return CalculateTNSR().get_documentation_arg_parser()


if __name__ == '__main__':
    CalculateTNSR().start()
