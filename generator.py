import tensorflow as tf
from time import time_ns

FLOAT = tf.float32
INT = tf.int32

def _get_seed():
    return [time_ns(), time_ns()+1000]

def normal(mean, std, shape=None):
    """
    Normal random varible generator.
    
    mean  : The mean of the normal distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    std   : The std of the normal distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    shape : Output shape. 
            If shape is None, the output shape will be broadcast_shape = tf.broadcast_static_shape(mean.shape, std.shape).
            If shape is given, the output shape will be tf.concat([shape, broadcast_shape], axis=0)
    """
    if not isinstance(mean, tf.Tensor):
        mean = tf.convert_to_tensor(mean, dtype=FLOAT)
    if not isinstance(std, tf.Tensor):
        std = tf.convert_to_tensor(std, dtype=FLOAT)
    broadcast_shape = tf.broadcast_static_shape(mean.shape, std.shape)
    if shape is None:
        shape = broadcast_shape
    else:
        shape = tf.concat([shape, broadcast_shape], axis=0)
    return tf.random.stateless_normal(shape, _get_seed(), mean, std)

def poisson(lam, shape=None):
    """
    Poisson random varible generator.
    
    lam   : The expectation of the Poisson distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    shape : Output shape. 
            If shape is None, the output shape will be lam.shape.
            If shape is given, the output shape will be tf.concat([shape, lam.shape], axis=0)
    """
    if not isinstance(lam, tf.Tensor):
        lam = tf.convert_to_tensor(lam, dtype=FLOAT)
    if shape is None:
        shape = lam.shape
    else:
        shape = tf.concat([shape, lam.shape], axis=0)
    return tf.random.stateless_poisson(shape, _get_seed(), lam)
    
def binomial(counts, probs, shape=None):
    """
    Binomial random varible generator.
    
    counts : The counts of the binomial distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    probs  : The probabilities of the binomial distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    shape  : Output shape. 
             If shape is None, the output shape will be broadcast_shape = tf.broadcast_static_shape(counts.shape, probs.shape).
             If shape is given, the output shape will be tf.concat([shape, broadcast_shape], axis=0)
    """
    if not isinstance(counts, tf.Tensor):
        counts = tf.convert_to_tensor(counts, dtype=FLOAT) # this is weird but if you use tf.int32 tf won't find a kernel
    if not isinstance(probs, tf.Tensor):
        probs = tf.convert_to_tensor(probs, dtype=FLOAT)
    if counts.dtype != FLOAT:
        counts = tf.cast(counts, FLOAT)
    probs = tf.clip_by_value(probs, 0, 1.)
    broadcast_shape = tf.broadcast_static_shape(counts.shape, probs.shape)
    if shape is None:
        shape = broadcast_shape
    else:
        shape = tf.concat([shape, broadcast_shape], axis=0)
    return tf.random.stateless_binomial(shape, _get_seed(), counts, probs)

def uniform(minval, maxval, shape=None):
    """
    Uniform random varible generator.
    
    minval : Lower bound of uniform distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    maxval : Upper bound of uniform distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    shape  : Output shape. 
             If shape is None, the output shape will be broadcast_shape = tf.broadcast_static_shape(minval.shape, maxval.shape).
             If shape is given, the output shape will be tf.concat([shape, broadcast_shape], axis=0)
    """
    if not isinstance(minval, tf.Tensor):
        minval = tf.convert_to_tensor(minval, dtype=FLOAT)
    if not isinstance(maxval, tf.Tensor):
        maxval = tf.convert_to_tensor(maxval, dtype=FLOAT)
    broadcast_shape = tf.broadcast_static_shape(minval.shape, maxval.shape)
    if shape is None:
        shape = broadcast_shape
    else:
        shape = tf.concat([shape, broadcast_shape], axis=0)
    return tf.random.stateless_uniform(shape, _get_seed(), minval, maxval)

def truncated_normal(mean, std, shape=None, vmin=None, vmax=None):
    """
    Truncated normal random varible generator.
    
    mean  : The mean of the normal distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    std   : The std of the normal distribution. Can be a single value, or tf.Tensor, or anything that can be converted to tf.Tensor.
    shape : Output shape. 
            If shape is None, the output shape will be broadcast_shape = tf.broadcast_static_shape(mean.shape, std.shape).
            If shape is given, the output shape will be tf.concat([shape, broadcast_shape], axis=0)
    vmin  : Lower truncation. Must be a signle value, and smaller than vmax.
    vmax  : Upper truncation. Must be a single value, and larger than vmin.
    """
    rv = normal(mean, std, shape)
    if (not vmin is None) and (not vmax is None):
        assert vmin < vmax, "vmin must be smaller than vmax!"
    if not vmin is None:
        rv = tf.where(rv<=vmin, vmin, rv)
    if not vmax is None:
        rv = tf.where(rv>=vmax, vmax, rv)
    return rv