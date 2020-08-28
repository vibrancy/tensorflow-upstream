# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.compat import collections_abc


def GetShrunkInceptionShapes(shrink=10):
  """Iterator for smaller versions of convolution shapes in 2015 Inception.

  Relative to inception, each depth value is `depth // shrink`.

  Args:
    shrink: Factor to shrink each depth value by relative to Inception.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the convolution
    parameters of Inception layers.
  """
  input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384],
                 [4, 8, 8, 2048], [4, 8, 8, 448], [4, 8, 8, 2048],
                 [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 1760],
                 [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1248],
                 [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1216],
                 [4, 17, 17, 1216], [4, 17, 17, 224], [4, 17, 17, 192],
                 [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152],
                 [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 1152],
                 [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024],
                 [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 128], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 768], [4, 35, 35, 96],
                 [4, 35, 35, 288], [4, 35, 35, 64], [4, 35, 35, 288],
                 [4, 35, 35, 256], [4, 35, 35, 48], [4, 35, 35, 256],
                 [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192],
                 [4, 35, 35, 192], [4, 73, 73, 64], [4, 73, 73, 64],
                 [4, 147, 147, 24]]
  filter_sizes = [[1, 1, 1248, 128], [1, 3, 384, 384], [3, 1, 384, 384],
                  [1, 1, 2048, 192], [3, 3, 448, 384], [1, 1, 2048, 320],
                  [1, 1, 2048, 448], [1, 1, 2048, 384], [1, 1, 1760, 384],
                  [1, 1, 1760, 192], [1, 1, 1760, 448], [1, 1, 1760, 320],
                  [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192],
                  [3, 3, 128, 320], [1, 1, 1248, 128], [1, 3, 224, 224],
                  [3, 1, 192, 256], [1, 3, 192, 256], [1, 1, 1216, 192],
                  [1, 1, 1216, 96], [3, 1, 224, 224], [3, 3, 192, 224],
                  [1, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128],
                  [3, 1, 192, 192], [3, 3, 160, 192], [1, 1, 1152, 160],
                  [1, 1, 1024, 128], [1, 3, 128, 192], [1, 1, 1024, 160],
                  [3, 1, 128, 192], [1, 1, 1024, 256], [3, 1, 128, 128],
                  [1, 1, 768, 192], [1, 3, 128, 128], [3, 3, 128, 128],
                  [1, 1, 768, 128], [1, 1, 768, 320], [3, 3, 96, 96],
                  [3, 3, 288, 384], [3, 3, 64, 96], [1, 1, 288, 64],
                  [1, 1, 256, 64], [5, 5, 48, 64], [1, 1, 256, 48],
                  [3, 3, 96, 96], [1, 1, 192, 32], [1, 1, 192, 64],
                  [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64, 64],
                  [1, 1, 24, 64]]
  out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 384], [4, 8, 8, 320],
               [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 448], [4, 8, 8, 320],
               [4, 8, 8, 192], [4, 17, 17, 192], [4, 17, 17, 192],
               [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224],
               [4, 17, 17, 256], [4, 17, 17, 256], [4, 17, 17, 192],
               [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 192], [4, 17, 17, 256], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128],
               [4, 17, 17, 128], [4, 17, 17, 320], [4, 17, 17, 96],
               [4, 17, 17, 384], [4, 35, 35, 96], [4, 35, 35, 64],
               [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48],
               [4, 35, 35, 96], [4, 35, 35, 32], [4, 35, 35, 64],
               [4, 35, 35, 48], [4, 71, 71, 192], [4, 73, 73, 64],
               [4, 147, 147, 64]]
  strides = [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1
  ]
  # Shrink sizes to make the test faster
  for i in input_sizes:
    i[3] //= shrink
  for f in filter_sizes:
    f[2] //= shrink
    f[3] //= shrink
  for o in out_sizes:
    o[3] //= shrink
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      VALID, SAME, SAME, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, VALID, VALID, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, VALID, VALID, VALID
  ]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCHW" format is only supported on CUDA.
    test_configs += [("NCHW", True)]
  return test_configs


class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    if use_gpu and not test_util.GpuSupportsHalfMatMulAndConv():
      return [dtypes.float32] + optional_float64
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return [dtypes.float32, dtypes.float16] + optional_float64

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, dilations,
                            strides, padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)

    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW(padding)
      conv = nn_ops.conv2d(
          t1,
          t2,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)
      self.assertEqual(conv.dtype, dtype)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    def _SetupVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        return conv

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-3, atol=1e-3)

  def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes,
                                   stride, dilation, padding, data_format,
                                   use_gpu):
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)
    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format)
      computed = nn_ops.conv2d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format)
      if data_format == "NCHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, strides,
                               padding, dilations, rtol=1e-4):
    expected_results = []
    computed_results = []
    for data_format, use_gpu in GetTestConfigs():
      expected, computed = self._ComputeReferenceDilatedConv(
          tensor_in_sizes, filter_in_sizes, strides, dilations, padding,
          data_format, use_gpu)
      expected_results.append(expected)
      computed_results.append(computed)
      tolerance = 1e-2 if use_gpu else 1e-5
      expected_values = self.evaluate(expected_results)
      computed_values = self.evaluate(computed_results)
      for e_value, c_value in zip(expected_values, computed_values):
        tf_logging.debug("expected = %s", e_value)
        tf_logging.debug("actual = %s", c_value)
        self.assertAllClose(
            e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=rtol)

  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    strides,
                    padding,
                    expected,
                    dilations=(1, 1),
                    gpu_only=False,
                    test_grappler_layout_optimizer=False,
                    tol=1e-5,
                    fp16_tol=1e-3):
    if gpu_only and not test.is_gpu_available(cuda_only=True):
      return
    tensors = []
    dilations = list(dilations)
    for (data_format, use_gpu) in GetTestConfigs():
      if gpu_only and not use_gpu:
        continue
      dtypes_to_test = self._DtypesToTest(use_gpu)
      if not test_grappler_layout_optimizer and data_format == "NHWC":
        dtypes_to_test.append(dtypes.int32)
      for dtype in dtypes_to_test:
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            dilations,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        if test_grappler_layout_optimizer and data_format == "NHWC" and use_gpu:
          # Grappler's layout optimizer will not optimize a fetch node, so
          # this identity allows Grappler to optimize the Conv2D node.
          result = array_ops.identity(result)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tf_logging.debug("expected = %s", expected)
        tf_logging.debug("actual = %s", value)
        tol_to_use = fp16_tol if value.dtype == np.float16 else tol
        if np.issubdtype(value.dtype, np.integer):
          self.assertAllEqual(np.rint(expected), np.ravel(value))
        else:
          self.assertAllClose(expected, np.ravel(value), atol=tol_to_use,
                              rtol=tol_to_use)
        self.assertShapeEqual(value, conv)
        self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)

  def _VerifyExplicitPaddings(self,
                              tensor_in_sizes,
                              filter_in_sizes,
                              strides,
                              padding,
                              dilations=(1, 1),
                              test_grappler_layout_optimizer=False,
                              tol=1e-5,
                              fp16_tol=1e-3):
    """Verifies Conv2D with explicit padding generates correct values.

    It does this by comparing with Conv2D without explicit padding. This
    function assumes Conv2D without explicit padding works correctly.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      strides: [row_stride, col_stride] for the convolution;
      padding: Explicit padding amounts.
      dilations: Dilation values
      test_grappler_layout_optimizer: If True, allow the Grappler layout
        optimizer to run, which turns NHWC Conv2Ds on the GPU to NCHW Conv2Ds.
      tol: The absolute and relative tolerance for non-fp16 dtypes.
      fp16_tol: The absolute and relative tolerance for fp16.
    """
    input_tensor = self._CreateNumpyTensor(tensor_in_sizes)
    filter_tensor = self._CreateNumpyTensor(filter_in_sizes)
    input_tensor = array_ops.pad(input_tensor, [(0, 0)] + padding + [(0, 0)])
    dilations = list(dilations)
    conv2d_result = nn_ops.conv2d(
        input_tensor,
        filter_tensor, [1] + list(strides) + [1],
        "VALID",
        dilations=[1] + dilations + [1])
    expected = list(self.evaluate(array_ops.reshape(conv2d_result, [-1])))
    self._VerifyValues(
        tensor_in_sizes,
        filter_in_sizes,
        strides,
        padding,
        expected,
        dilations,
        test_grappler_layout_optimizer=test_grappler_layout_optimizer,
        tol=tol,
        fp16_tol=fp16_tol)


  def _VerifyGroupConvFwd(self, tensor_in_sizes, filter_in_sizes, dilations,
                          strides, padding, data_format, dtype):
    """Verify the output of group convolution is equal to a for-loop implementation.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
    """
    tensor_in = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    num_groups = tensor_in_sizes[3] // filter_in_sizes[2]
    assert num_groups > 1 and \
        filter_in_sizes[2] * num_groups == tensor_in_sizes[3]
    with test_util.device(True):
      t1 = constant_op.constant(tensor_in, dtype=dtype)
      t2 = constant_op.constant(filter_in, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        t1_splits = array_ops.split(t1, num_groups, axis=1)
      else:
        t1_splits = array_ops.split(t1, num_groups, axis=3)
      t2_splits = array_ops.split(t2, num_groups, axis=3)

      def MakeConv2d(inputs, filters):
        return nn_ops.conv2d(
            inputs,
            filters,
            strides,
            padding,
            dilations=dilations,
            data_format=data_format)

      group_conv = MakeConv2d(t1, t2)
      group_conv_loop = array_ops.concat(
          [MakeConv2d(t1s, t2s) for t1s, t2s in zip(t1_splits, t2_splits)],
          axis=1 if data_format == "NCHW" else 3)

      results = self.evaluate([group_conv, group_conv_loop])
      tol_to_use = 1e-5
      self.assertAllClose(
          results[0], results[1], atol=tol_to_use, rtol=tol_to_use)

  # TODO(yzhwang): this currently fails.
  # self._VerifyValues(tensor_in_sizes=[1, 8, 8, 1],
  #                   filter_in_sizes=[2, 2, 1, 1],
  #                   strides=[4, 4], padding="SAME",
  #                   expected=[72, 112, 392, 432])

  # Testing for backprops
  def _RunAndVerifyBackpropInput(self,
                                 input_sizes,
                                 filter_sizes,
                                 output_sizes,
                                 strides,
                                 padding,
                                 expected,
                                 data_format,
                                 use_gpu,
                                 err,
                                 dilations=(1, 1)):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    with test_util.device(use_gpu):
      if len(input_sizes) == 4:
        if data_format == "NCHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
      t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
      t1 = constant_op.constant(x1, shape=filter_sizes)
      t2 = constant_op.constant(x2, shape=output_sizes)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t2 = test_util.NHWCToNCHW(t2)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW((padding))
      conv = nn_ops.conv2d_backprop_input(
          t0,
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)
      # "values" consists of two tensors for two backprops
      value = self.evaluate(conv)
      self.assertShapeEqual(value, conv)
    tf_logging.debug("expected = %s", expected)
    tf_logging.debug("actual = %s", value)
    self.assertAllCloseAccordingToType(expected, value.flatten(), atol=1e-5)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            conv_strides, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        if data_format == "NCHW":
          new_input_sizes = test_util.NHWCToNCHW(input_sizes)
        else:
          new_input_sizes = input_sizes
        t0 = constant_op.constant(new_input_sizes, shape=[len(new_input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t2 = test_util.NHWCToNCHW(t2)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_input(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        ret = self.evaluate(conv)
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for (data_format, use_gpu) in GetTestConfigs():
      values.append(_GetVal(data_format, use_gpu))

    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-2, atol=1e-2)

  # Testing for backprops
  def _RunAndVerifyBackpropFilter(self,
                                  input_sizes,
                                  filter_sizes,
                                  output_sizes,
                                  strides,
                                  padding,
                                  expected,
                                  data_format,
                                  use_gpu,
                                  dilations=(1, 1),
                                  err=1e-5):
    x0 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    explicit_strides = [1] + strides + [1]
    new_padding = padding
    new_dilations = [1] + dilations + [1]
    if isinstance(new_padding, (list, tuple)):
      new_padding = [(0, 0)] + new_padding + [(0, 0)]
    if data_format == "NCHW":
      explicit_strides = test_util.NHWCToNCHW(explicit_strides)
      new_dilations = test_util.NHWCToNCHW(new_dilations)
      if isinstance(padding, (list, tuple)):
        new_padding = test_util.NHWCToNCHW(new_padding)
    for dtype in self._DtypesToTest(use_gpu=use_gpu):
      with test_util.device(use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        if data_format == "NCHW":
          t0 = test_util.NHWCToNCHW(t0)
          t2 = test_util.NHWCToNCHW(t2)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=explicit_strides,
            padding=new_padding,
            dilations=new_dilations,
            data_format=data_format)
        value = self.evaluate(conv)
        self.assertShapeEqual(value, conv)
      tf_logging.debug("expected = %s", expected)
      tf_logging.debug("actual = %s", value)
      self.assertArrayNear(expected, value.flatten(), err)

  def _CompareBackFilter(self, input_sizes, filter_sizes, output_sizes,
                         conv_strides, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t0 = test_util.NHWCToNCHW(t0)
          t2 = test_util.NHWCToNCHW(t2)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        ret = self.evaluate(conv)
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for (data_format, use_gpu) in GetTestConfigs():
      values.append(_GetVal(data_format, use_gpu))
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-4, atol=1e-4)

  # Testing for backprops
  def _RunAndVerifyBackpropInputDilation(self, input_sizes, filter_sizes,
                                         output_sizes, strides, dilations,
                                         padding, data_format, use_gpu, err):
    x1 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(filter_sizes)
    default_dilations = (dilations[0] == 1 and dilations[1] == 1)
    if default_dilations or use_gpu:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if data_format == "NCHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
        t1 = constant_op.constant(x1, shape=input_sizes)
        t2 = constant_op.constant(x2, shape=filter_sizes)
        full_strides = [1] + strides + [1]
        full_dilations = [1] + dilations + [1]
        if data_format == "NCHW":
          full_strides = test_util.NHWCToNCHW(full_strides)
          full_dilations = test_util.NHWCToNCHW(full_dilations)
        conv_forward = nn_ops.conv2d(
            t1,
            t2,
            strides=full_strides,
            dilations=full_dilations,
            padding=padding,
            data_format=data_format)
        conv_forward_2 = nn_ops.convolution(
            t1,
            t2,
            padding=padding,
            strides=strides,
            dilation_rate=dilations,
            data_format=data_format)
        if data_format == "NCHW":
          conv_forward = test_util.NCHWToNHWC(conv_forward)
          conv_forward_2 = test_util.NCHWToNHWC(conv_forward_2)
        conv = gradients_impl.gradients(conv_forward, t1)[0]
        conv_2 = gradients_impl.gradients(conv_forward_2, t1)[0]
        # "values" consists of two tensors for two backprops
        value = self.evaluate(conv)
        value_2 = self.evaluate(conv_2)
        self.assertShapeEqual(value, conv)
        self.assertShapeEqual(value_2, conv_2)
      tf_logging.debug("expected = %s", value_2)
      tf_logging.debug("actual = %s", value)
      self.assertArrayNear(value_2.flatten(), value.flatten(), err)

  # Testing for backprops
  def _RunAndVerifyBackpropFilterDilation(self, input_sizes, filter_sizes,
                                          output_sizes, strides, dilations,
                                          padding, data_format, use_gpu, err):
    x1 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(filter_sizes)
    default_dilations = (dilations[0] == 1 and dilations[1] == 1)
    if default_dilations or use_gpu:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if data_format == "NCHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
        t1 = constant_op.constant(x1, shape=input_sizes)
        t2 = constant_op.constant(x2, shape=filter_sizes)
        full_strides = [1] + strides + [1]
        full_dilations = [1] + dilations + [1]
        if data_format == "NCHW":
          full_strides = test_util.NHWCToNCHW(full_strides)
          full_dilations = test_util.NHWCToNCHW(full_dilations)
        conv_forward = nn_ops.conv2d(
            t1,
            t2,
            strides=full_strides,
            dilations=full_dilations,
            padding=padding,
            data_format=data_format)
        conv_forward_2 = nn_ops.convolution(
            t1,
            t2,
            padding=padding,
            strides=strides,
            dilation_rate=dilations,
            data_format=data_format)
        if data_format == "NCHW":
          conv_forward = test_util.NCHWToNHWC(conv_forward)
          conv_forward_2 = test_util.NCHWToNHWC(conv_forward_2)
        conv = gradients_impl.gradients(conv_forward, t2)[0]
        conv_2 = gradients_impl.gradients(conv_forward, t2)[0]
        value = self.evaluate(conv)
        value_2 = self.evaluate(conv_2)
        self.assertShapeEqual(value, conv)
        self.assertShapeEqual(value_2, conv_2)
      tf_logging.debug("expected = %s", value_2)
      tf_logging.debug("actual = %s", value)
      self.assertArrayNear(value_2.flatten(), value.flatten(), err)


  def _RunAndVerifyBackpropInputExplicitPadding(self,
                                                input_sizes,
                                                filter_sizes,
                                                output_sizes,
                                                strides,
                                                padding,
                                                data_format,
                                                use_gpu,
                                                dilations=(1, 1),
                                                err=2e-5):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    if not use_gpu and dilations != (1, 1):
      return  # Non-default dilations is currently not supported on the CPU.

    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    padded_input_sizes = input_sizes[:]
    padded_input_sizes[1] += padding[0][0] + padding[0][1]
    padded_input_sizes[2] += padding[1][0] + padding[1][1]
    c = nn_ops.conv2d_backprop_input(
        padded_input_sizes,
        x1,
        x2,
        strides=[1] + strides + [1],
        padding="VALID",
        dilations=[1] + dilations + [1])
    c = c[:, padding[0][0]:(c.shape[1] - padding[0][1]), padding[1][0]:(
        c.shape[2] - padding[1][1]), :]
    expected = list(self.evaluate(array_ops.reshape(c, [-1])))
    self._RunAndVerifyBackpropInput(
        input_sizes,
        filter_sizes,
        output_sizes,
        strides,
        padding,
        expected,
        data_format,
        use_gpu=use_gpu,
        err=err,
        dilations=dilations)


  def _RunAndVerifyBackpropFilterExplicitPadding(self,
                                                 input_sizes,
                                                 filter_sizes,
                                                 output_sizes,
                                                 strides,
                                                 padding,
                                                 data_format,
                                                 use_gpu,
                                                 dilations=(1, 1),
                                                 err=1e-5):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    if not use_gpu and dilations != (1, 1):
      return  # Non-default dilations is currently not supported on the CPU.

    x0 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)

    x0 = np.pad(x0, [(0, 0)] + padding + [(0, 0)], "constant")
    c = nn_ops.conv2d_backprop_filter(
        x0,
        filter_sizes,
        x2,
        strides=[1] + strides + [1],
        padding="VALID",
        dilations=[1] + dilations + [1])
    expected = list(self.evaluate(array_ops.reshape(c, [-1])))
    self._RunAndVerifyBackpropFilter(
        input_sizes,
        filter_sizes,
        output_sizes,
        strides,
        padding,
        expected,
        data_format,
        use_gpu=use_gpu,
        dilations=dilations,
        err=err)

  # Gradient checkers
  def ConstructAndTestGradient(self,
                               batch,
                               input_rows,
                               input_cols,
                               filter_rows,
                               filter_cols,
                               in_depth,
                               out_depth,
                               stride_rows,
                               stride_cols,
                               padding,
                               test_input,
                               data_format,
                               use_gpu,
                               num_groups=1,
                               max_err=0.003):
    assert in_depth % num_groups == 0 and out_depth % num_groups == 0
    input_shape = [batch, input_rows, input_cols, in_depth]
    filter_shape = [filter_rows, filter_cols, in_depth // num_groups, out_depth]
    # TODO(yangke): re-factor the computation of output shape.
    if padding == "VALID":
      output_rows = (input_rows - filter_rows + stride_rows) // stride_rows
      output_cols = (input_cols - filter_cols + stride_cols) // stride_cols
    elif padding == "SAME":
      output_rows = (input_rows + stride_rows - 1) // stride_rows
      output_cols = (input_cols + stride_cols - 1) // stride_cols
    else:
      self.assertIsInstance(padding, (list, tuple))
      output_rows = (input_rows + padding[1][0] + padding[1][1] - filter_rows +
                     stride_rows) // stride_rows
      output_cols = (input_cols + padding[2][0] + padding[2][1] - filter_cols +
                     stride_cols) // stride_cols
    output_shape = [batch, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    # Conv2DGrad functions are not compiled for double due to
    # a problem in the way Eigen's Conv2DGrad works for double.
    # So we disable the DOUBLE path.  We should re-enable this
    # when double support returns for CPU and/or GPU.
    for dtype in self._DtypesToTest(use_gpu=use_gpu):
    # for dtype in [dtypes.float16]:
      print (dtype)
      with self.cached_session(use_gpu=use_gpu):
        input_tensor = constant_op.constant(
            input_data, shape=input_shape, dtype=dtype, name="input")
        filter_tensor = constant_op.constant(
            filter_data, shape=filter_shape, dtype=dtype, name="filter")
        strides = [1, stride_rows, stride_cols, 1]
        new_padding = padding
        if data_format == "NCHW":
          new_input_tensor = test_util.NHWCToNCHW(input_tensor)
          strides = test_util.NHWCToNCHW(strides)
          if isinstance(padding, (list, tuple)):
            new_padding = test_util.NHWCToNCHW(padding)
        else:
          new_input_tensor = input_tensor
        conv = nn_ops.conv2d(
            new_input_tensor,
            filter_tensor,
            strides,
            new_padding,
            data_format=data_format,
            name="conv")
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        self.assertEqual(output_shape, conv.get_shape())
        if test_input:
          jacob_t, jacob_n = gradient_checker.compute_gradient(input_tensor,
                                                               input_shape,
                                                               conv,
                                                               output_shape)
        else:
          jacob_t, jacob_n = gradient_checker.compute_gradient(filter_tensor,
                                                               filter_shape,
                                                               conv,
                                                               output_shape)
        if dtype == dtypes.float32:
          reference_jacob_t = jacob_t
          err = np.fabs(jacob_t - jacob_n).max()
        else:
          # Compare fp16 theoretical gradients to fp32 theoretical gradients,
          # since fp16 numerical gradients are too imprecise.
          err = np.fabs(jacob_t - reference_jacob_t).max()

        tf_logging.debug("conv_2d gradient error = %s", err)
        self.assertLess(err, max_err)

  @test_util.deprecated_graph_mode_only
  def testFilterGradientSamePaddingStrideOne(self):
    # for (data_format, use_gpu) in GetTestConfigs():
    for (data_format, use_gpu) in [('NHWC', 'True')]:
      print (data_format, use_gpu)
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)



if __name__ == "__main__":
  test.main()
