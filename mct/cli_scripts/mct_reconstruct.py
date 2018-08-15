#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Reconstruct your images using the desired method.
"""
import argparse
import os
from argcomplete.completers import FilesCompleter
from mot.lib import cl_environments

import mct
from mct.components_loader import list_reconstruction_methods
from mdt.lib.nifti import nifti_filepath_resolution
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Reconstruct(BasicShellApplication):

    def __init__(self):
        super().__init__()
        self.available_devices = list((ind for ind, env in
                                       enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection())))

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-reconstruct rSoS {0..15}.nii.gz
            mct-reconstruct STARC {0..15}.nii -o ./output
            mct-reconstruct STARC {0..15}.nii --kwargs starting_points=weights.nii
            mct-reconstruct STARC {0..15}.nii --cl-device-ind {0, 1}
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('method_name', type=str,
                            help='the method to use for the reconstruction')

        parser.add_argument('input_files', type=str, nargs='+',
                            help='the input channels')

        parser.add_argument('-o', '--output_dir', default=None,
                            help='the output directory, defaults to a subdir in the dir '
                                 'of the first weight.').completer = FilesCompleter()

        parser.add_argument('-v', '--volumes', required=False, nargs='+',
                            help='the volumes to use in the reconstruction, either a list of indices '
                                 'or the literal "odd", "even" or "all"')

        parser.add_argument('--cl-device-ind', type=int, nargs='*', choices=self.available_devices,
                            help="The index of the device we would like to use. This follows the indices "
                                 "in mdt-list-devices and defaults to the first GPU.")

        parser.add_argument('--kwargs', type=str, nargs='+',
                            help='Optional keyword arguments for the model, provide as <key>=<value> pairs,'
                                 'see the model for which arguments are supported.')
        return parser

    def run(self, args, extra_args):
        if args.method_name not in list_reconstruction_methods():
            raise ValueError('The given model {} can not be found.'.format(args.method_name))

        method_kwargs = get_keyword_args(args.kwargs, os.path.realpath(''))
        method_kwargs['cl_device_ind'] = args.cl_device_ind

        input_files = get_input_files(args.input_files, os.path.realpath(''))

        if not input_files:
            raise ValueError('No coils provided, please provide at least one coil.')

        if args.output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_files[0]), 'output')
        else:
            output_dir = os.path.realpath(args.output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        volumes_to_use = None
        if args.volumes:
            if args.volumes[0] == 'odd':
                volumes_to_use = 'odd'
            elif args.volumes[0] == 'even':
                volumes_to_use = 'even'
            elif args.volumes[0] == 'all':
                volumes_to_use = None
            else:
                volumes_to_use = list(map(int, args.volumes))

        method = mct.load_reconstruction_method(args.method_name, input_files, **method_kwargs)
        method.reconstruct(output_dir, volumes=volumes_to_use)


def get_keyword_args(kwargs, base_dir):
    keyword_args = {}

    if kwargs:
        for argument in kwargs:
            key, value = argument.split('=')

            if os.path.isfile(value):
                keyword_args[key] = value
            elif os.path.isfile(base_dir + '/' + value):
                keyword_args[key] = base_dir + '/' + value
            else:
                keyword_args[key] = value

    return keyword_args


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
    return Reconstruct().get_documentation_arg_parser()


if __name__ == '__main__':
    Reconstruct().start()
