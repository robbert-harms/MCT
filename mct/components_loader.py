from mct.reconstruction_methods import STARC, rSoS, rCovSoS

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


reconstruction_classes = [STARC, rSoS, rCovSoS]


def list_reconstruction_methods():
    """Get a list of the reconstruction methods by name.

    Returns:
        list[str]: the list of reconstruction methods
    """
    return [cls.__name__ for cls in reconstruction_classes]


def load_reconstruction_method(method_name, *args, **kwargs):
    """Load the requested reconstruction method as an object.


    Args:
        method_name (str): the name of the reconstruction method to load
        *args: passed to the constructor of the requested method
        **kwargs: passed to the constructor of the requested method

    Returns:
        mct.processing.ReconstructionMethod: the class of the requested reconstruction method

    Raises:
        ValueError: if the requested method could not be found
    """
    method = get_reconstruction_method_class(method_name)
    return method(*args, **kwargs)


def get_reconstruction_method_class(method_name):
    """Load the class of the requested reconstruction method without instantiation.

    Args:
        method_name (str): the name of the reconstruction method to load

    Returns:
        type[mct.processing.ReconstructionMethod]: the class of the requested reconstruction method

    Raises:
        ValueError: if the requested method could not be found
    """
    for method in reconstruction_classes:
        if method.__name__ == method_name:
            return method
    raise ValueError('The requestion reconstruction method ({}) could not be found.'.format(method_name))
