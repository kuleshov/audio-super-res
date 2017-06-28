"""Class for doing iterations over datasets

This is stolen from the tensorflow tutorial
"""

import numpy

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# ----------------------------------------------------------------------------

class DataSet(object):

  def __init__(self,
               datapoints,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    if labels is None:
      labels = np.zeros((len(datapoints),))

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert datapoints.shape[0] == labels.shape[0], (
          'datapoints.shape: %s labels.shape: %s' % (datapoints.shape, labels.shape))
      self._num_examples = datapoints.shape[0]

    self._datapoints = datapoints
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def datapoints(self):
    return self._datapoints

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._datapoints = self.datapoints[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      datapoints_rest_part = self._datapoints[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._datapoints = self.datapoints[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      datapoints_new_part = self._datapoints[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((datapoints_rest_part, datapoints_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._datapoints[start:end], self._labels[start:end]
