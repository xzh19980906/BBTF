import tensorflow as tf

def set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def set_device_log_on():
    tf.debugging.set_log_device_placement(True)
    
def set_device_log_off():
    tf.debugging.set_log_device_placement(False)
        
def get_devices():
    return tf.config.list_physical_devices()

