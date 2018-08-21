import logging.config as logging_config
import yaml
from mct.components_loader import load_reconstruction_method
from mct.utils import get_mot_config_context
from .__version__ import VERSION, VERSION_STATUS, __version__
from mct.utils import load_nifti, combine_weighted_sum
from mct.components_loader import load_reconstruction_method, get_reconstruction_method_class
from mct.reconstruction_methods import rCovSoS, rSoS, STARC

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"


try:
    config = '''
        version: 1
        disable_existing_loggers: False

        formatters:
            simple:
                format: "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s"

        handlers:
            console:
                class: mdt.lib.log_handlers.StdOutHandler
                level: INFO
                formatter: simple

            dispatch_handler:
                class: mdt.lib.log_handlers.LogDispatchHandler
                level: INFO
                formatter: simple

        loggers:
            mct:
                level: DEBUG
                handlers: [console]

        root:
            level: INFO
            handlers: [dispatch_handler]
    '''
    logging_config.dictConfig(yaml.safe_load(config))

except ValueError as ex:
    print('Logging disabled: {}'.format(ex))
