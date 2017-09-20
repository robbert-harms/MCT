from textwrap import dedent

import six
from mot.model_building.parameter_functions.transformations import CosSqrClampTransform

from mct.utils import calculate_tsnr
from mot.model_building.model_builders import ParameterTransformedModel

from mct.processing import ReconstructionMethod
from mot import Powell
import mdt
import numpy as np

from mot.model_building.utils import ParameterCodec
from mot.model_interfaces import OptimizeModelInterface
from mot.utils import dtype_to_ctype, KernelInputBuffer, SimpleNamedCLFunction

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class STARC(ReconstructionMethod):

    command_line_info = dedent('''
        The STARC (STAbility-weighted Rf-coil Combination) method [1] reconstructs EPI acquisitions using a weighted sum of the input channels. The weights are chosen such that the reconstruction has optimal tSNR.
        
        Required args:
            None
        
        Optional keyword args:
            starting_points="<nifti_file>" - the starting point for the optimization routine
        
        References:
            * Simple approach to improve time series fMRI stability: STAbility-weighted Rf-coil Combination (STARC), L. Huber et al. ISMRM 2017 abstract #0586.
    ''')

    def __init__(self, *args, starting_points=None, **kwargs):
        """Reconstruct the input using the STARC method.

        Args:
            starting_points (ndarray or str): optional, the set of weights to use as a starting point
                for the fitting routine.
        """
        self._starting_points = starting_points
        if isinstance(self._starting_points, six.string_types):
            self._starting_points = mdt.load_nifti(starting_points).get_data()

    def reconstruct(self, batch, volume_indices):
        self._optimizer = Powell(patience=2)

        starting_weights = None
        if self._starting_points is not None:
            starting_weights = self._starting_points[np.split(volume_indices, 3, axis=1)]
            starting_weights = np.reshape(starting_weights, (-1, batch.shape[-1]))

        model = STARCModel(batch, starting_weights=starting_weights)
        bounded_model = ParameterTransformedModel(model, model.get_parameter_codec())

        result_struct = self._optimizer.minimize(bounded_model)
        del starting_weights

        weights = result_struct.get_optimization_result()
        reconstruction = np.sum(batch * weights[:, None, :], axis=2)

        return {'weights': weights, 'reconstruction': reconstruction, 'tSNR': calculate_tsnr(reconstruction)}


class STARCModel(OptimizeModelInterface):

    def __init__(self, voxel_data, starting_weights=None):
        """Create the STARC model such that MOT can fit it.

        This model maximizes the tSNR (``tSNR = mean(time_series') / std(time_series')``) by minimizing 1/tSNR, or, in
        other words by minimizing ``std(time_series') / mean(time_series')`` where ``time_series'`` is given by
        the weighted sum of the provided time series over the channels.

        The model parameters are a set of weights with, for every voxel, one weight per coil element. The weights are
        constrained to be between [0, 1] as such that the sum of the weights equals one.

        Model output is an ndarray (nmr_voxels, nmr_channels) holding the optimized weights for each of the voxels.

        Args:
            voxel_data (ndarray): a 3d matrix with (nmr_voxels, nmr_volumes, nmr_channels).
            starting_weights (ndarray): a 2d matrix with (nmr_voxels, nmr_channels), if provided, it is the
                set of starting weights for the optimization routine.
        """
        self.voxel_data = voxel_data
        self.starting_weights = starting_weights
        self.nmr_voxels = voxel_data.shape[0]
        self.nmr_volumes = voxel_data.shape[1]
        self.nmr_channels = voxel_data.shape[2]
        self._data_ctype = dtype_to_ctype(self.voxel_data.dtype)

        if self.starting_weights is not None:
            if len(self.starting_weights.shape) != 2:
                raise ValueError('The starting weights should have exactly '
                                 'two dimensions, {} given.'.format(len(self.starting_weights.shape)))
            if self.starting_weights.shape[1] != self.voxel_data.shape[2]:
                raise ValueError('The number of channels of the input data ({}) '
                                 'and the starting weights ({}) should match.'.format(self.voxel_data.shape[2],
                                                                                      self.starting_weights.shape[1]))

    @property
    def name(self):
        return 'weighted_coils'

    @property
    def double_precision(self):
        return False

    def get_kernel_data(self):
        return {'observations': KernelInputBuffer(self.voxel_data.reshape((self.nmr_voxels, -1)))}

    def get_nmr_problems(self):
        return self.nmr_voxels

    def get_nmr_inst_per_problem(self):
        # returns the inverse of the tSNR as the only observation instance
        return 1

    def get_nmr_estimable_parameters(self):
        return self.nmr_channels

    def get_pre_eval_parameter_modifier(self):
        # we always need to return a placeholder
        func_name = '_preEvaluation'
        func = 'void ' + func_name + '(mot_data_struct* data, const mot_float_type* x){}'
        return SimpleNamedCLFunction(func, func_name)

    def get_model_eval_function(self):
        pass  # not needed

    def get_residual_per_observation_function(self):
        pass  # not needed

    def get_objective_per_observation_function(self):
        fname = '_objectiveFunc'
        func = '''
            double _weighted_sum(mot_data_struct* data, const mot_float_type* const x, uint volume_index){

                global ''' + self._data_ctype + '''* observations = data->observations;

                double sum = 0;
                for(uint i = 0; i < ''' + str(self.nmr_channels) + '''; i++){
                    sum += x[i] * observations[i + volume_index * ''' + str(self.nmr_channels) + '''];
                }                
                return sum;
            }

            double _inverse_tSNR(mot_data_struct* data, const mot_float_type* x){
                double variance = 0;
                double mean = 0;
                double delta;
                double value;
                for(uint i = 0; i < ''' + str(self.nmr_volumes) + '''; i++){
                    value = _weighted_sum(data, x, i);
                    delta = value - mean;
                    mean += delta / (i + 1);
                    variance += delta * (value - mean);
                }
                variance /= (''' + str(self.nmr_volumes) + ''' - 1);
                return sqrt(variance) / mean;
            }

            double ''' + fname + '''(mot_data_struct* data, const mot_float_type* const x, uint observation_index){
                return _inverse_tSNR(data, x);
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def get_initial_parameters(self):
        if self.starting_weights is not None:
            return self.starting_weights
        return np.ones((self.nmr_voxels, self.nmr_channels)) / float(self.nmr_channels)

    def get_lower_bounds(self):
        return np.zeros((self.nmr_voxels, self.nmr_channels))

    def get_upper_bounds(self):
        return np.ones((self.nmr_voxels, self.nmr_channels))

    def finalize_optimized_parameters(self, parameters):
        return parameters

    def get_parameter_codec(self):
        """Create a parameter codec to enforce the boundary conditions.

        This will limit every weight between [0, 1] and makes sure that the weights sum to one.
        """
        model_builder = self
        param_codec = CosSqrClampTransform()

        class Codec(ParameterCodec):
            def get_parameter_decode_function(self, function_name='decodeParameters'):
                decode_transform = param_codec.get_cl_decode()
                func = '''
                    void ''' + function_name + '''(mot_data_struct* data_void, mot_float_type* x){
                        double sum_of_weights = 0;
                        for(uint i = 0; i < ''' + str(model_builder.nmr_channels) + '''; i++){
                            x[i] = ''' + decode_transform.create_assignment('x[i]', 0, 1) + ''';    
                            sum_of_weights += x[i];
                        }

                        for(uint i = 0; i < ''' + str(model_builder.nmr_channels) + '''; i++){
                            x[i] = x[i] / sum_of_weights;
                        }
                    } 
                '''
                return func

            def get_parameter_encode_function(self, function_name='encodeParameters'):
                encode_transform = param_codec.get_cl_encode()
                func = '''
                    void ''' + function_name + '''(mot_data_struct* data_void, mot_float_type* x){
                        double sum_of_weights = 0;
                        for(uint i = 0; i < ''' + str(model_builder.nmr_channels) + '''; i++){    
                            sum_of_weights += x[i];
                        }

                        for(uint i = 0; i < ''' + str(model_builder.nmr_channels) + '''; i++){
                            x[i] = x[i] / sum_of_weights;
                        }

                        for(uint i = 0; i < ''' + str(model_builder.nmr_channels) + '''; i++){
                            x[i] = ''' + encode_transform.create_assignment('x[i]', 0, 1) + ''';
                        }
                    } 
                '''
                return func

        return Codec()
