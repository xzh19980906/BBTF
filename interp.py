import tensorflow as tf
import tensorflow_probability as tfp

class KNN():
    def __init__(self, points, values, k=3):
        """
        points : tensor-like with shape (N, D), as the reference points in D-dim
        values : tensor-like with shape (N,), as the values at reference points
        k      : int, as the number of the nearest neighbors
        
        The algorithm is the same as the one in straxen (https://github.com/XENONnT/straxen/blob/a8402c46053105ad5c60b38ae91ef9b8a1571aa7/straxen/itp_map.py#L15).
        """
        self.ref_points = tf.convert_to_tensor(points, dtype=tf.float32)
        self.ref_values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.k = k
        self.num_ref, self.dim = tf.shape(points)
        
        assert self.num_ref > self.k, "Number of reference points must be larger than k!"
        assert self.num_ref == len(self.ref_values), "Shape of points %s is inconsistent with shape of values %s!"%(self.ref_points.shape, self.ref_values.shape)
        
    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)
        
    def _L2_dist2(self, pop1, pop2):
        """
        pop1 : tensor-like with shape (N, D)
        pop2 : tensor-like with shape (M, D)
        
        return : L2 distance squared with shape (N, M)
        """
        dr = tf.expand_dims(pop1, axis=1) - tf.expand_dims(pop2, axis=0)
        return tf.reduce_sum(tf.square(dr), axis=-1)
        
    def interp(self, points):
        """
        points : tensor-like with shape (N, D), as the points to be interpolated.
        
        return : interpolated values at the points with shape (N,), weighted by the inverse of distance to k nearest neighbors. 
        """
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        
        dr2 = - self._L2_dist2(points, self.ref_points)
        dr2, ind = tf.math.top_k(dr2, self.k)
        weights = 1.0 / tf.clip_by_value(tf.sqrt(-dr2), 1e-6, float('inf'))
        vals = tf.gather(self.ref_values, ind)
        vals = tf.math.reduce_sum(vals * weights, axis=1) / tf.math.reduce_sum(weights, axis=1)
        return vals
    
    
class NN_LinearGrid():
    def __init__(self, values, binning):
        """
        values  : tensor-like with shape (ax1_n_points, ..., axD_n_points)
        binning : [
            [ax1_lower, ax1_upper, ax1_n_points], 
            ...,
            [axD_lower, axD_upper, axD_n_points], 
            ]
            
        The algorithm is the same as the one in straxen with k=2^dim (https://github.com/XENONnT/straxen/blob/a8402c46053105ad5c60b38ae91ef9b8a1571aa7/straxen/itp_map.py#L15).
        """
        self.ref_values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.binning = binning
        self.dim = len(binning)
        
        self.lowers = tf.constant([axis[0] for axis in binning], dtype=tf.float32)
        self.stepsize = tf.constant([(axis[1]-axis[0])/(axis[2]-1) for axis in binning], dtype=tf.float32)
        self.num_bins = tf.constant([(axis[2]-1) for axis in binning], dtype=tf.int32)
        self.lowers = tf.reshape(self.lowers, (1, 1, self.dim))
        self.stepsize = tf.reshape(self.stepsize, (1, 1, self.dim))
        self.num_bins = tf.reshape(self.num_bins, (1, 1, self.dim))
        
        self._magic_ind_shift = tf.transpose(tf.constant([([1]*(2**i)+[0]*(2**i))*(2**(self.dim-i-1)) for i in range(self.dim)], dtype=tf.int32))
        self._magic_ind_shift = tf.expand_dims(self._magic_ind_shift, axis=1)
        
        assert len(binning) == len(tf.shape(values)), "binning (%i-D) must have the same dimension as values (%i-D)!"%(len(binning), len(tf.shape(values)))
        assert tf.reduce_all(tf.shape(values) == [axis[2] for axis in binning]), "values shape is not consistent with binning!"
    
    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)
    
    def _get_nn_indices(self, points):
        """
        points : point positions with shape (1, N, D)
        
        return : 2^D nearest neighbor indices with shape (2^D, N, D)
        """
        ind = tf.clip_by_value(tf.cast((points - self.lowers) / self.stepsize, tf.int32), 0, self.num_bins-1)
        return ind + self._magic_ind_shift
    
    def interp(self, points):
        """
        points : tensor-like with shape (N, D), as the points to be interpolated.
        
        return : interpolated values at the points with shape (N,), weighted by the inverse of distance to 2^D nearest neighbors. 
        """
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        
        points = tf.expand_dims(points, axis=0)
        ind = self._get_nn_indices(points)
        ref_pos = self.lowers + self.stepsize * tf.cast(ind, tf.float32)
        dr2 = tf.reduce_sum(tf.square(points - ref_pos), axis=-1)
        weights = 1. / tf.clip_by_value(tf.sqrt(dr2), 1e-6, float('inf'))
        vals = tf.gather_nd(self.ref_values, ind)
        vals = tf.math.reduce_sum(vals * weights, axis=0) / tf.math.reduce_sum(weights, axis=0)
        return vals
    
    
class Linear1D():
    def __init__(self, points, values):
        """
        points : tensor-like with shape (N,), as the reference points
        values : tensor-like with shape (N,), as the values at reference points
        """
        self.ref_points = tf.convert_to_tensor(points, dtype=tf.float32)
        self.ref_values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.num_ref = len(points)
        
        order = tf.argsort(self.ref_points)
        self.ref_points = tf.gather(self.ref_points, order)
        self.ref_values = tf.gather(self.ref_values, order)
        
        assert self.num_ref > 2, "Number of reference points must be larger than 2!"
        
    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)
        
    def _dist(self, pop1, pop2):
        """
        pop1 : tensor-like with shape (N,)
        pop2 : tensor-like with shape (M,)
        
        return : L2 distance squared with shape (N, M)
        """
        dr = tf.expand_dims(pop1, axis=1) - tf.expand_dims(pop2, axis=0)
        return dr
        
    def interp(self, points):
        """
        points : tensor-like with shape (N,), as the points to be interpolated.
        
        return : linearly interpolated values at the points with shape (N,). Extration is clipped by the boundary values. 
        """
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        
        dr = self._dist(points, self.ref_points)
        ind = tf.reduce_sum(tf.cast(dr>=0, dtype=tf.int32), axis=-1) # find right ref point
        ind = tf.stack((ind-1, ind), axis=1) # stack with left ref point
        ind = tf.clip_by_value(ind, 0, self.num_ref-1) # clip if outside
        
        vals = tf.gather(self.ref_values, ind)
        rpos = tf.gather(self.ref_points, ind)
        
        x0 = rpos[:, 0]
        x1 = rpos[:, 1]
        y0 = vals[:, 0]
        y1 = vals[:, 1]
        dx = x1 - x0 # dx could be zero if extrap
        y = ((x1 - points) * y0 + (points - x0) * y1) / dx
        
        return tf.where(dx==0, tf.gather(self.ref_values, ind[:, 0]), y)
    
    
class Linear1D_LinearGrid():
    def __init__(self, values, binning):
        """
        values  : tensor-like with shape (N,), as the reference values
        binning : [lower, upper, n_points]
        """
        self.ref_values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.lower, self.upper, self.n_points = binning
    
    def __call__(self, *args, **kwargs):
        return self.interp(*args, **kwargs)
    
    def interp(self, points):
        """
        points : tensor-like with shape (N,), as the points to be interpolated.
        
        return : linearly interpolated values at the points with shape (N,). Extration is clipped by the boundary values. 
        """
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        return tfp.math.interp_regular_1d_grid(points, self.lower, self.upper, self.ref_values, fill_value='constant_extension')