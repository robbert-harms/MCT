#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""List the available reconstruction methods.

This lists the methods that are available in the mct-reconstruct command.

To view more information about a method use the mct-method-info command.
"""
import argparse
from mct.components_loader import list_reconstruction_methods
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2017-09-09"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ListMethods(BasicShellApplication):

    def __init__(self):
        super().__init__()

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mct-list-methods
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        return parser

    def run(self, args, extra_args):
        for method_name in list_reconstruction_methods():
            print(method_name)


def get_doc_arg_parser():
    return ListMethods().get_documentation_arg_parser()


if __name__ == '__main__':
    ListMethods().start()
