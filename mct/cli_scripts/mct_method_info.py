#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Lists information about a method.

This outputs the documentation of the desired method.
"""
import argparse
from mct import get_reconstruction_method_class
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MethodInfo(BasicShellApplication):

    def __init__(self):
        super().__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-method-info rSoS        
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('method_name', type=str, help='the name of the model you want the info of')
        return parser

    def run(self, args, extra_args):
        cls = get_reconstruction_method_class(args.method_name)
        print(cls.command_line_info)


def get_doc_arg_parser():
    return MethodInfo().get_documentation_arg_parser()


if __name__ == '__main__':
    MethodInfo().start()
