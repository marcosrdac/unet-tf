from typing import Optional, Union, List, Tuple, Any
import itertools
import six
import tensorflow as tf

def _check_type(variable, variable_name, expected_type):
    """Helper function for checking that inputs are of expected types."""
    if isinstance(expected_type, (list, tuple)):
        expected_type_name = 'list or tuple'
    else:
        expected_type_name = expected_type.__name__
    if not isinstance(variable, expected_type):
        raise ValueError('{} must be of type {}, but it is {}'.format(
            variable_name, expected_type_name,
            type(variable).__name__))
    
def _raise_value_error_for_rank(variable, error_msg):
    raise ValueError(
        '{} must have a rank {} {}, but it has rank {} and shape {}'.format(
            tensor_name, error_msg, variable, rank, tensor.shape.as_list()))

def _fix_axis_dim_pairs(pairs, name):
  """Helper function to make `pairs` a list if needed."""
  if isinstance(pairs[0], int):
    pairs = [pairs]
  for pair in pairs:
    if len(pair) != 2:
      raise ValueError(
          '{} must consist of axis-value pairs, but found {}'.format(
              name, pair))
  return pairs

def _get_dim(tensor, axis):
  """Returns dimensionality of a tensor for a given axis."""
  return tf.compat.dimension_value(tensor.shape[axis])

def compare_batch_dimensions(
    tensors: Union[List[tf.Tensor], Tuple[tf.Tensor]],
    last_axes: Union[int, List[int], Tuple[int]],
    broadcast_compatible: bool,
    initial_axes: Union[int, List[int], Tuple[int]] = 0,
    tensor_names: Optional[Union[List[str], Tuple[str]]] = None) -> None:
  """Compares batch dimensions for tensors with static shapes.
  Args:
    tensors: A list or tuple of tensors with static shapes to compare.
    last_axes: An `int` or a list or tuple of `int`s with the same length as
      `tensors`. If an `int`, it is assumed to be the same for all the tensors.
      Each entry should correspond to the last axis of the batch (with zero
      based indices). For instance, if there is only a single batch dimension,
      last axis should be `0`.
    broadcast_compatible: A 'bool', whether the batch shapes can be broadcast
      compatible in the numpy sense.
    initial_axes: An `int` or a list or tuple of `int`s with the same length as
      `tensors`. If an `int`, it is assumed to be the same for all the tensors.
      Each entry should correspond to the first axis of the batch (with zero
      based indices). Default value is `0`.
    tensor_names: Names of `tensors` to be used in the error message if one is
      thrown. If left as `None`, `tensor_i` is used.
  Raises:
    ValueError: If inputs have unexpected types, or if given axes are out of
      bounds, or if the check fails.
  """
  _check_tensors(tensors, 'tensors')
  if isinstance(initial_axes, int):
    initial_axes = [initial_axes] * len(tensors)
  if isinstance(last_axes, int):
    last_axes = [last_axes] * len(tensors)
  _check_tensor_axis_lists(tensors, 'tensors', initial_axes, 'initial_axes')
  _check_tensor_axis_lists(tensors, 'tensors', last_axes, 'last_axes')
  initial_axes = _fix_axes(tensors, initial_axes, allow_negative=True)
  last_axes = _fix_axes(tensors, last_axes, allow_negative=True)
  batch_shapes = [
      tensor.shape[init:last + 1]
      for tensor, init, last in zip(tensors, initial_axes, last_axes)
  ]
  if tensor_names is None:
    tensor_names = _give_default_names(tensors, 'tensor')
  if not broadcast_compatible:
    batch_ndims = [batch_shape.ndims for batch_shape in batch_shapes]
    batch_shapes = [batch_shape.as_list() for batch_shape in batch_shapes]
    if not _all_are_equal(batch_ndims):
      # If not all batch shapes have the same length, they cannot be identical.
      _raise_error(tensor_names, batch_shapes)
    for dims in zip(*batch_shapes):
      if _all_are_equal(dims):
        # Continue if all dimensions are None or have the same value.
        continue
      if None not in dims:
        # If all dimensions are known at this point, they are not identical.
        _raise_error(tensor_names, batch_shapes)
      # At this point dims must consist of both None's and int's.
      if len(set(dims)) != 2:
        # set(dims) should return (None, some_int).
        # Otherwise shapes are not identical.
        _raise_error(tensor_names, batch_shapes)
  else:
    if not all(
        is_broadcast_compatible(shape1, shape2)
        for shape1, shape2 in itertools.combinations(batch_shapes, 2)):
      raise ValueError(
          'Not all batch dimensions are broadcast-compatible: {}'.format([
              (name, batch_shape.as_list())
              for name, batch_shape in zip(tensor_names, batch_shapes)
          ]))
    
def _check_tensors(tensors, tensors_name):
  """Helper function to check the type and length of tensors."""
  _check_type(tensors, tensors_name, (list, tuple))
  if len(tensors) < 2:
    raise ValueError('At least 2 tensors are required.')
    
def _check_tensor_axis_lists(tensors, tensors_name, axes, axes_name):
  """Helper function to check that lengths of `tensors` and `axes` match."""
  _check_type(axes, axes_name, (list, tuple))
  if len(tensors) != len(axes):
    raise ValueError(
        '{} and {} must have the same length, but are {} and {}.'.format(
            tensors_name, axes_name, len(tensors), len(axes)))

def _fix_axes(tensors, axes, allow_negative):
  """Makes all axes positive and checks for out of bound errors."""
  axes = [
      axis + tensor.shape.ndims if axis < 0 else axis
      for tensor, axis in zip(tensors, axes)
  ]
  if not all(
      ((allow_negative or
        (not allow_negative and axis >= 0)) and axis < tensor.shape.ndims)
      for tensor, axis in zip(tensors, axes)):
    rank_axis_pairs = list(
        zip([tensor.shape.ndims for tensor in tensors], axes))
    raise ValueError(
        'Some axes are out of bounds. Given rank-axes pairs: {}'.format(
            [pair for pair in rank_axis_pairs]))
  return axes
    
def is_broadcast_compatible(shape_x: tf.TensorShape,
                            shape_y: tf.TensorShape) -> bool:
  """Returns True if `shape_x` and `shape_y` are broadcast compatible.
  Args:
    shape_x: A `TensorShape`.
    shape_y: A `TensorShape`.
  Returns:
    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted
    to. False otherwise.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return False
  return _broadcast_shape_helper(shape_x, shape_y) is not None

def _broadcast_shape_helper(shape_x: tf.TensorShape,
                            shape_y: tf.TensorShape) -> Optional[List[Any]]:
  """Helper function for is_broadcast_compatible and broadcast_shape.
  Args:
    shape_x: A `TensorShape`.
    shape_y: A `TensorShape`.
  Returns:
    Returns None if the shapes are not broadcast compatible, or a list
    containing the broadcasted dimensions otherwise.
  """
  # To compute the broadcasted dimensions, we zip together shape_x and shape_y,
  # and pad with 1 to make them the same length.
  broadcasted_dims = reversed(
      list(
          six.moves.zip_longest(
              reversed(shape_x.dims),
              reversed(shape_y.dims),
              fillvalue=tf.compat.v1.Dimension(1))))
  # Next we combine the dimensions according to the numpy broadcasting rules.
  # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      # One or both dimensions is unknown. If either dimension is greater than
      # 1, we assume that the program is correct, and the other dimension will
      # be broadcast to match it.
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      # We will broadcast dim_x to dim_y.
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      # We will broadcast dim_y to dim_x.
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      # The dimensions are compatible, so output is the same size in that
      # dimension.
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      return None
  return return_dims
    
def check_static(tensor: tf.Tensor,
                 has_rank: Optional[int] = None,
                 has_rank_greater_than: Optional[int] = None,
                 has_rank_less_than: Optional[int] = None,
                 has_dim_equals=None,
                 has_dim_greater_than=None,
                 has_dim_less_than=None,
                 tensor_name: str = 'tensor') -> None:
    """Checks static shapes for rank and dimension constraints.
    This function can be used to check a tensor's shape for multiple rank and
    dimension constraints at the same time.
    Args:
    tensor: Any tensor with a static shape.
    has_rank: An int or `None`. If not `None`, the function checks if the rank
      of the `tensor` equals to `has_rank`.
    has_rank_greater_than: An int or `None`. If not `None`, the function checks
      if the rank of the `tensor` is greater than `has_rank_greater_than`.
    has_rank_less_than: An int or `None`. If not `None`, the function checks if
      the rank of the `tensor` is less than `has_rank_less_than`.
    has_dim_equals: Either a tuple or list containing a single pair of `int`s,
      or a list or tuple containing multiple such pairs. Each pair is in the
      form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] == dim`.
    has_dim_greater_than: Either a tuple or list containing a single pair of
      `int`s, or a list or tuple containing multiple such pairs. Each pair is in
      the form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] > dim`.
    has_dim_less_than: Either a tuple or list containing a single pair of
      `int`s, or a list or tuple containing multiple such pairs. Each pair is in
      the form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] < dim`.
    tensor_name: A name for `tensor` to be used in the error message if one is
      thrown.
    Raises:
    ValueError: If any input is not of the expected types, or if one of the
      checks described above fails.
    """
    rank = tensor.shape.ndims

    def _raise_value_error_for_rank(variable, error_msg):
        raise ValueError(
            '{} must have a rank {} {}, but it has rank {} and shape {}'.format(
                tensor_name, error_msg, variable, rank, tensor.shape.as_list()))
    
    def _raise_value_error_for_dim(tensor_name, error_msg, axis, value):
        raise ValueError(
            '{} must have {} {} dimensions in axis {}, but it has shape {}'.format(
                tensor_name, error_msg, value, axis, tensor.shape.as_list()))
  
    if has_rank is not None:
        _check_type(has_rank, 'has_rank', int)
        if rank != has_rank:
            _raise_value_error_for_rank(has_rank, 'of')
    if has_rank_greater_than is not None:
        _check_type(has_rank_greater_than, 'has_rank_greater_than', int)
        if rank <= has_rank_greater_than:
            _raise_value_error_for_rank(has_rank_greater_than, 'greater than')
    if has_rank_less_than is not None:
        _check_type(has_rank_less_than, 'has_rank_less_than', int)
        if rank >= has_rank_less_than:
            _raise_value_error_for_rank(has_rank_less_than, 'less than')
    if has_dim_equals is not None:
        _check_type(has_dim_equals, 'has_dim_equals', (list, tuple))
        has_dim_equals = _fix_axis_dim_pairs(has_dim_equals, 'has_dim_equals')
        for axis, value in has_dim_equals:
            if _get_dim(tensor, axis) != value:
                _raise_value_error_for_dim(tensor_name, 'exactly', axis, value)
    if has_dim_greater_than is not None:
        _check_type(has_dim_greater_than, 'has_dim_greater_than', (list, tuple))
        has_dim_greater_than = _fix_axis_dim_pairs(has_dim_greater_than,
                                                   'has_dim_greater_than')
        for axis, value in has_dim_greater_than:
            if not _get_dim(tensor, axis) > value:
                _raise_value_error_for_dim(tensor_name, 'greater than', axis, value)
    if has_dim_less_than is not None:
        _check_type(has_dim_less_than, 'has_dim_less_than', (list, tuple))
        has_dim_less_than = _fix_axis_dim_pairs(has_dim_less_than,
                                                'has_dim_less_than')
        for axis, value in has_dim_less_than:
            if not _get_dim(tensor, axis) < value:
                _raise_value_error_for_dim(tensor_name, 'less than', axis, value)

def trilinear_interpolate(grid_3d,
                sampling_points,
                name: str = "trilinear_interpolate"):
  """Trilinear interpolation on a 3D regular grid.
  Args:
    grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
      height, width, depth of the grid and C is the number of channels.
    sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
      number of sampling points. Sampling points outside the grid are projected
      in the grid borders.
    name:  A name for this op that defaults to "trilinear_interpolate".
  Returns:
    A tensor of shape `[A1, ..., An, M, C]`
  """

  with tf.name_scope(name):
    grid_3d = tf.convert_to_tensor(value=grid_3d)
    sampling_points = tf.convert_to_tensor(value=sampling_points)

    check_static(
        tensor=grid_3d, tensor_name="grid_3d", has_rank_greater_than=3)
    check_static(
        tensor=sampling_points,
        tensor_name="sampling_points",
        has_dim_equals=(-1, 3),
        has_rank_greater_than=1)
    compare_batch_dimensions(
        tensors=(grid_3d, sampling_points),
        last_axes=(-5, -3),
        tensor_names=("grid_3d", "sampling_points"),
        broadcast_compatible=True)
    voxel_cube_shape = tf.shape(input=grid_3d)[-4:-1]
    sampling_points.set_shape(sampling_points.shape)
    batch_dims = tf.shape(input=sampling_points)[:-2]
    num_points = tf.shape(input=sampling_points)[-2]

    bottom_left = tf.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = tf.cast(bottom_left, tf.int32)
    top_right_index = tf.cast(top_right, tf.int32)
    x0_index, y0_index, z0_index = tf.unstack(bottom_left_index, axis=-1)
    x1_index, y1_index, z1_index = tf.unstack(top_right_index, axis=-1)
    index_x = tf.concat([x0_index, x1_index, x0_index, x1_index,
                         x0_index, x1_index, x0_index, x1_index], axis=-1)
    index_y = tf.concat([y0_index, y0_index, y1_index, y1_index,
                         y0_index, y0_index, y1_index, y1_index], axis=-1)
    index_z = tf.concat([z0_index, z0_index, z0_index, z0_index,
                         z1_index, z1_index, z1_index, z1_index], axis=-1)
    indices = tf.stack([index_x, index_y, index_z], axis=-1)
    clip_value = tf.convert_to_tensor(
        value=[voxel_cube_shape - 1], dtype=indices.dtype)
    indices = tf.clip_by_value(indices, 0, clip_value)
    content = tf.gather_nd(
        params=grid_3d, indices=indices, batch_dims=tf.size(input=batch_dims))
    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = tf.unstack(distance_to_bottom_left, axis=-1)
    x1_x, y1_y, z1_z = tf.unstack(distance_to_top_right, axis=-1)
    weights_x = tf.concat([x1_x, x_x0, x1_x, x_x0,
                           x1_x, x_x0, x1_x, x_x0], axis=-1)
    weights_y = tf.concat([y1_y, y1_y, y_y0, y_y0,
                           y1_y, y1_y, y_y0, y_y0], axis=-1)
    weights_z = tf.concat([z1_z, z1_z, z1_z, z1_z,
                           z_z0, z_z0, z_z0, z_z0], axis=-1)
    weights = tf.expand_dims(weights_x * weights_y * weights_z, axis=-1)

    interpolated_values = weights * content
    return tf.add_n(tf.split(interpolated_values, [num_points] * 8, -2))


def trilinear_resample(tensor, shape):
    idx = tf.cast(tf.where(tf.ones(shape[:-1], dtype=tf.uint8)), dtype=tf.float32)
    idx *= (tf.cast(tensor.shape[:-1], dtype=tf.float32) - 1) / (tf.cast(shape[:-1], tf.float32) - 1)
    return tf.reshape(trilinear_interpolate(tensor, idx), shape)
