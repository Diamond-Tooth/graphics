#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the iou loss metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow_graphics.nn.losses.iou as iou
from tensorflow_graphics.util import test_case


class IOUTest(test_case.TestCase):

  @parameterized.parameters(
      # 1 dimensional grid.
      ([1, 0, 0, 1, 1, 0, 1], \
       [1, 0, 1, 1, 1, 1, 0], 3. / 6.),
      # 2 dimensional grid.
      ([[1, 0, 1], [0, 0, 1], [0, 1, 1]], \
       [[0, 1, 1], [1, 1, 1], [0, 0, 1]], 3. / 8.),
      ([[0, 0, 0], [0, 0, 0]], \
       [[1, 1, 0], [0, 0, 1]], 0.),
      # Returns 1 on empty union.
      ([0, 0, 0], [0, 0, 0], 1.),
  )
  def test_compute_loss_preset(self, ground_truth, predictions, expected_iou):
    tensor_size = np.random.randint(5) + 1
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()

    grid_size = np.array(ground_truth).ndim
    ground_truth_labels = np.tile(ground_truth, tensor_shape + [1] * grid_size)
    predicted_labels = np.tile(predictions, tensor_shape + [1] * grid_size)
    first_grid_axis = -grid_size
    expected = np.tile(expected_iou, tensor_shape)

    result = iou.compute_loss(ground_truth_labels, predicted_labels,
                              first_grid_axis)

    self.assertAllClose(expected, result)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (1, 5, 3), (4, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 4), (2, 4, 5)),
  )
  def test_estimate_radiance_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(iou.compute_loss, error_msg, shape)
