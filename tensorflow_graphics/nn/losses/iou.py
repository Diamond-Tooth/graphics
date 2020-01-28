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
"""This module implements the iou loss metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def compute_loss(ground_truth_labels,
                 predicted_labels,
                 first_grid_axis=-1,
                 name=None):
  """Computes the Intersection-Over-Union loss metric for the given ground truth and predicted labels.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
    G1 to Gm are the grid dimensions.

  Args:
    ground_truth_labels: A tensor of shape '[A1, ..., An, G1, ..., Gm]', where
      the last m axes represent a grid of boolean ground truth labels.
    predicted_labels: A tensor of shape '[A1, ..., An, G1, ..., Gm]', where the
      last m axes represent a grid of boolean predictions.
    first_grid_axis: The index of the first grid axis. Defaults to -1.
    name: A name for this op. Defaults to "compute_iou_loss".

  Returns:
    A tensor of shape `[A1, ..., An]`, where the last axis represents the
    computed iou loss of the given ground truth labels and predictions.

  Raises:
    ValueError: if the shape of `ground_truth_labels`, `predicted_labels` is
    not supported.
  """
  with tf.compat.v1.name_scope(
      name, "compute_iou_loss",
      [ground_truth_labels, predicted_labels, first_grid_axis]):
    ground_truth_labels = tf.convert_to_tensor(value=ground_truth_labels)
    predicted_labels = tf.convert_to_tensor(value=predicted_labels)

    shape.compare_batch_dimensions(
        tensors=(ground_truth_labels, predicted_labels),
        tensor_names=("ground_truth_labels", "predicted_labels"),
        last_axes=-1,
        broadcast_compatible=True)

    intersection = tf.math.minimum(ground_truth_labels, predicted_labels)
    union = tf.math.maximum(ground_truth_labels, predicted_labels)
    # Flatten the grid dimensions.
    intersection = tf.reshape(
        intersection, intersection.shape[:first_grid_axis].as_list() + [-1])
    union = tf.reshape(union, union.shape[:first_grid_axis].as_list() + [-1])
    # Compute the intersection and union size.
    intersection_size = tf.math.reduce_sum(input_tensor=intersection, axis=-1)
    union_size = tf.math.reduce_sum(input_tensor=union, axis=-1)

    common_shape = shape.get_broadcasted_shape(intersection_size.shape,
                                               union_size.shape)
    d_val = lambda dim: 1 if dim is None else tf.compat.v1.dimension_value(dim)
    common_shape = [d_val(dim) for dim in common_shape]
    # Return 1 if the union is empty, otherwise the intersection-union ratio.
    return tf.compat.v1.where(
        tf.broadcast_to(tf.math.equal(union_size, 0), common_shape),
        tf.ones_like(union_size, dtype=tf.float64),
        intersection_size / union_size)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
