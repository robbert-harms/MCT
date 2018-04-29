from textwrap import dedent

import six

from mct.utils import get_cl_devices
from mdt.model_building.parameter_functions.transformations import CosSqrClampTransform
from mdt.model_building.utils import ParameterTransformedModel

from mct.reconstruction import SliceBySliceReconstructionMethod
from mot import Powell
import mdt
import numpy as np

from mdt.model_building.utils import ParameterCodec
from mot.cl_runtime_info import CLRuntimeInfo
from mot.model_interfaces import OptimizeModelInterface
from mot.utils import dtype_to_ctype, NameFunctionTuple
from mot.kernel_data import KernelArray

__author__ = 'Robbert Harms'
__date__ = '2017-09-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class STARC(SliceBySliceReconstructionMethod):

    command_line_info = dedent('''
        The STARC (STAbility-weighted Rf-coil Combination) method [1] reconstructs EPI acquisitions using a weighted sum of the input channels. The weights are chosen such that the reconstruction has optimal tSNR.

        Required args:
            None

        Optional keyword args:
            starting_points="<nifti_file>" - the starting point for the optimization routine

        References:
            * Simple approach to improve time series fMRI stability: STAbility-weighted Rf-coil Combination (STARC), L. Huber et al. ISMRM 2017 abstract #0586.
    ''')

    def __init__(self, channels, starting_points=None, cl_device_ind=None, **kwargs):
        """Reconstruct the input using the STARC method.

        Args:
            channels (list): the list of input nifti files, one for each channel element. Every nifti file
                    should be a 4d matrix with on the 4th dimension all the time series. The length of this list
                    should equal the number of input channels.
            starting_points (ndarray or str): optional, the set of weights to use as a starting point
                for the fitting routine.
            cl_device_ind (list of int): the list of indices into :func:`mct.utils.get_cl_devices` that you want
                to use for the OpenCL based optimization.
        """
        super(STARC, self).__init__(channels, **kwargs)

        cl_environments = None
        if cl_device_ind is not None:
            if not isinstance(cl_device_ind, (tuple, list)):
                cl_device_ind = [cl_device_ind]
            cl_environments = [get_cl_devices()[ind] for ind in cl_device_ind]

        self._optimizer = Powell(patience=2, cl_runtime_info=CLRuntimeInfo(cl_environments=cl_environments))
        self._starting_points = starting_points
        if isinstance(self._starting_points, six.string_types):
            self._starting_points = mdt.load_nifti(starting_points).get_data()

    def _reconstruct_slice(self, slice_data, slice_index):
        nmr_timeseries = slice_data.shape[-2]
        nmr_channels = slice_data.shape[-1]

        batch = np.reshape(slice_data, (-1, nmr_timeseries, nmr_channels))

        constrained_model = ParameterTransformedModel(STARCModel(batch), STARCOptimizationCodec(nmr_channels))

        result_struct = self._optimizer.minimize(
            constrained_model, constrained_model.encode_parameters(self._get_starting_weights(slice_index)))

        weights = result_struct.get_optimization_result()
        reconstruction = np.sum(batch * weights[:, None, :], axis=2)

        sos = np.sqrt(np.sum(np.abs(slice_data).astype(np.float64) ** 2, axis=-1))
        reconstruction = np.reshape(reconstruction, slice_data.shape[:-2] + (nmr_timeseries,))
        reconstruction *= (np.mean(sos, axis=2) / np.mean(reconstruction, axis=2))[:, :, None]

        return {
            'weights': np.reshape(weights, slice_data.shape[:-2] + (nmr_channels,)),
            'reconstruction': reconstruction,
        }

    def _get_starting_weights(self, slice_index):
        starting_weights = None
        if self._starting_points is not None:
            index = [slice(None)] * 3
            index[self._slicing_axis] = slice_index
            starting_weights = self._starting_points[index]
            starting_weights = np.reshape(starting_weights, (-1, starting_weights.shape[-1]))
        return starting_weights


class STARCModel(OptimizeModelInterface):

    def __init__(self, voxel_data):
        """Create the STARC model such that MOT can fit it.

        This model maximizes the tSNR (``tSNR = mean(time_series') / std(time_series')``) by minimizing 1/tSNR, or, in
        other words by minimizing ``std(time_series') / mean(time_series')`` where ``time_series'`` is given by
        the weighted sum of the provided time series over the channels.

        The model parameters are a set of weights with, for every voxel, one weight per coil element. The weights are
        constrained to be between [0, 1] as such that the sum of the weights equals one.

        Model output is an ndarray (nmr_voxels, nmr_channels) holding the optimized weights for each of the voxels.

        Args:
            voxel_data (ndarray): a 3d matrix with (nmr_voxels, nmr_volumes, nmr_channels).
        """
        self.voxel_data = voxel_data
        self.nmr_voxels = voxel_data.shape[0]
        self.nmr_volumes = voxel_data.shape[1]
        self.nmr_channels = voxel_data.shape[2]
        self._data_ctype = dtype_to_ctype(self.voxel_data.dtype)

    def get_kernel_data(self):
        return {'observations': KernelArray(self.voxel_data.reshape((self.nmr_voxels, -1)))}

    def get_nmr_problems(self):
        return self.nmr_voxels

    def get_nmr_observations(self):
        # returns the inverse of the tSNR as the only observation instance
        return 1

    def get_nmr_parameters(self):
        return self.nmr_channels

    def get_objective_per_observation_function(self):
        fname = '_objectiveFunc'
        func = '''
            double _weighted_sum(mot_float_type* weights, global ''' + self._data_ctype + '''* observations){
                double sum = 0;
                for(uint i = 0; i < ''' + str(self.nmr_channels) + '''; i++){
                    sum += weights[i] * observations[i];
                }
                return sum;
            }

            double _inverse_tSNR(mot_data_struct* data, const mot_float_type* x){
                double variance = 0;
                double mean = 0;
                double delta;
                double value;
                for(uint i = 0; i < ''' + str(self.nmr_volumes) + '''; i++){
                    value = _weighted_sum(x, data->observations + i * ''' + str(self.nmr_channels) + ''');
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
        return NameFunctionTuple(fname, func)

    def get_lower_bounds(self):
        return np.zeros((self.nmr_voxels, self.nmr_channels))

    def get_upper_bounds(self):
        return np.ones((self.nmr_voxels, self.nmr_channels))

    def finalize_optimized_parameters(self, parameters):
        return parameters


class STARCOptimizationCodec(ParameterCodec):

    def __init__(self, nmr_optimized_weights):
        """Create a parameter codec to enforce the boundary conditions.

        Parameter codecs are a optimization trick to enforce boundary conditions and to (in some cases) present a
        smoother optimization landscape to the optimization routine. Before optimization the parameters are transformed
        from model space to optimization space which is what the optimizer will iteratively try to improve. Just before
        model evaluation, each point suggested by the optimizer will be decoded back into model space.

        Both the model as well as the optimization routine do not need to know that this transformation takes place.

        This particular codec limits every weight between [0, 1] and makes sure that the weights sum to one.

        Args:
            nmr_optimized_weights (int): the number of weights we are optimizing
        """
        self._nmr_optimized_weights = nmr_optimized_weights
        self._weights_codec = CosSqrClampTransform()

    def get_parameter_decode_function(self, function_name='decodeParameters'):
        decode_transform = self._weights_codec.get_cl_decode()
        func = '''
            void ''' + function_name + '''(mot_data_struct* data_void, mot_float_type* x){
                double sum_of_weights = 0;
                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = ''' + decode_transform.create_assignment('x[i]', 0, 1) + ''';
                    sum_of_weights += x[i];
                }

                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = x[i] / sum_of_weights;
                }
            }
        '''
        return func

    def get_parameter_encode_function(self, function_name='encodeParameters'):
        encode_transform = self._weights_codec.get_cl_encode()
        func = '''
            void ''' + function_name + '''(mot_data_struct* data_void, mot_float_type* x){
                double sum_of_weights = 0;
                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    sum_of_weights += x[i];
                }

                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = x[i] / sum_of_weights;
                }

                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = ''' + encode_transform.create_assignment('x[i]', 0, 1) + ''';
                }
            }
        '''
        return func
