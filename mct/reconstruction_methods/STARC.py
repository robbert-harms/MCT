from textwrap import dedent

from mct.utils import get_cl_devices
from mdt.model_building.parameter_functions.transformations import CosSqrClampTransform
from mdt.model_building.utils import ParameterDecodingWrapper

from mct.reconstruction import SliceBySliceReconstructionMethod
import mdt
import numpy as np

from mdt.model_building.utils import ParameterCodec
from mot import minimize
from mot.lib.cl_function import SimpleCLFunction
from mot.configuration import CLRuntimeInfo
from mot.lib.utils import dtype_to_ctype, parse_cl_function
from mot.lib.kernel_data import Array, Struct, LocalMemory

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

    def __init__(self, channels, x0=None, cl_device_ind=None, **kwargs):
        """Reconstruct the input using the STARC method.

        Args:
            channels (list): the list of input nifti files, one for each channel element. Every nifti file
                    should be a 4d matrix with on the 4th dimension all the time series. The length of this list
                    should equal the number of input channels.
            x0 (ndarray or str): optional, the set of weights to use as a starting point for the fitting routine.
            cl_device_ind (int or list of int): the list of indices into :func:`mct.utils.get_cl_devices` that you want
                to use for the OpenCL based optimization.
        """
        super().__init__(channels, **kwargs)

        cl_environments = None
        if cl_device_ind is not None:
            if not isinstance(cl_device_ind, (tuple, list)):
                cl_device_ind = [cl_device_ind]
            cl_environments = [get_cl_devices()[ind] for ind in cl_device_ind]

        self.cl_runtime_info = CLRuntimeInfo(cl_environments=cl_environments)
        self._x0 = x0
        if isinstance(self._x0, str):
            self._x0 = mdt.load_nifti(x0).get_data()

    def _reconstruct_slice(self, slice_data, slice_index):
        nmr_timeseries = slice_data.shape[-2]
        nmr_channels = slice_data.shape[-1]

        batch = np.reshape(slice_data, (-1, nmr_timeseries, nmr_channels))

        codec = STARCOptimizationCodec(nmr_channels)

        data = Struct({'observations': Array(batch.reshape((batch.shape[0], -1))),
                       'scratch': LocalMemory('double', nmr_items=batch.shape[1] + 4)},
                      'starc_data')

        wrapper = ParameterDecodingWrapper(nmr_channels)

        result = minimize(wrapper.wrap_objective_function(get_starc_objective_func(batch), codec.get_decode_function()),
                          codec.encode(self._get_starting_weights(slice_index, batch)),
                          data=wrapper.wrap_input_data(data),
                          cl_runtime_info=self.cl_runtime_info)

        weights = codec.decode(result['x'])
        reconstruction = np.sum(batch * weights[:, None, :], axis=2)

        sos = np.sqrt(np.sum(np.abs(slice_data).astype(np.float64) ** 2, axis=-1))
        reconstruction = np.reshape(reconstruction, slice_data.shape[:-2] + (nmr_timeseries,))
        reconstruction *= (np.mean(sos, axis=2) / np.mean(reconstruction, axis=2))[:, :, None]

        return {
            'weights': np.reshape(weights, slice_data.shape[:-2] + (nmr_channels,)),
            'reconstruction': reconstruction,
        }

    def _get_starting_weights(self, slice_index, current_batch):
        if self._x0 is None:
            nmr_voxels = current_batch.shape[0]
            nmr_channels = current_batch.shape[2]
            return np.ones((nmr_voxels, nmr_channels)) / float(nmr_channels)

        index = [slice(None)] * 3
        index[self._slicing_axis] = slice_index
        starting_weights = self._x0[index]
        return np.reshape(starting_weights, (-1, starting_weights.shape[-1]))


def get_starc_objective_func(voxel_data):
    """Create the STARC objective function used by MOT.

    This model maximizes the tSNR (``tSNR = mean(time_series') / std(time_series')``) by minimizing 1/tSNR, or, in
    other words by minimizing ``std(time_series') / mean(time_series')`` where ``time_series'`` is given by
    the weighted sum of the provided time series over the channels.

    The model parameters are a set of weights with, for every voxel, one weight per coil element. The weights are
    constrained to be between [0, 1] as such that the sum of the weights equals one.

    Model output is an ndarray (nmr_voxels, nmr_channels) holding the optimized weights for each of the voxels.

    Args:
        voxel_data (ndarray): a 3d matrix with (nmr_voxels, nmr_volumes, nmr_channels).
    """
    nmr_volumes = voxel_data.shape[1]
    nmr_channels = voxel_data.shape[2]
    data_ctype = dtype_to_ctype(voxel_data.dtype)

    return parse_cl_function('''
        double _weighted_sum(local const mot_float_type* weights, global ''' + data_ctype + '''* observations){
            double sum = 0;
            for(uint i = 0; i < ''' + str(nmr_channels) + '''; i++){
                sum += weights[i] * observations[i];
            }
            return sum;
        }

        double _inverse_tSNR(
                local const mot_float_type* x, 
                global ''' + data_ctype + '''* observations,
                local double* scratch){
            
            local double* variance = scratch++;
            local double* mean = scratch++;
            local double* delta = scratch++;
            local double* value = scratch++;    
            local double* volume_values = scratch;
            
            uint local_id = get_local_id(0);
            uint workgroup_size = get_local_size(0);

            uint volume_ind;
            for(uint i = 0; i < (''' + str(nmr_volumes) + ''' + workgroup_size - 1) / workgroup_size; i++){
                volume_ind = i * workgroup_size + local_id;
                
                if(volume_ind < ''' + str(nmr_volumes) + '''){
                    volume_values[volume_ind] = _weighted_sum(
                        x, observations + volume_ind * ''' + str(nmr_channels) + ''');
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(get_local_id(0) == 0){
                *variance = 0;
                *mean = 0;

                for(uint i = 0; i < ''' + str(nmr_volumes) + '''; i++){
                    *value = volume_values[i];
                    *delta = *value - *mean;
                    *mean += *delta / (i + 1);
                    *variance += *delta * (*value - *mean);
                }
                *variance /= (''' + str(nmr_volumes) + ''' - 1);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return sqrt(*variance) / *mean;
        }
        
        double STARC(local const mot_float_type* const x, void* data, local mot_float_type* objective_list){
            return _inverse_tSNR(x, ((starc_data*)data)->observations, ((starc_data*)data)->scratch);
        }
    ''')


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
        super().__init__(self._get_encode_function(), self._get_decode_function())

    def _get_encode_function(self):
        encode_transform = self._weights_codec.get_cl_encode()
        return SimpleCLFunction.from_string('''
            void encodeParameters(void* data, local mot_float_type* x){
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
        ''')

    def _get_decode_function(self):
        decode_transform = self._weights_codec.get_cl_decode()
        return SimpleCLFunction.from_string('''
            void decodeParameters(void* data, local mot_float_type* x){
                double sum_of_weights = 0;
                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = ''' + decode_transform.create_assignment('x[i]', 0, 1) + ''';
                    sum_of_weights += x[i];
                }

                for(uint i = 0; i < ''' + str(self._nmr_optimized_weights) + '''; i++){
                    x[i] = x[i] / sum_of_weights;
                }
            }
        ''')
