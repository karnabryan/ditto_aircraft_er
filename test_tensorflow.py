import tensorflow as tf

# Print TensorFlow version
#print("TensorFlow version:", tf.__version__)


build_info = tf.sysconfig.get_build_info()
print("CUDA Version:", build_info.get("cuda_version"))
print("CUDNN Version:", build_info.get("cudnn_version"))

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("No GPU found. Please check your CUDA setup.")

# Optional: Enable device placement logging to see where operations are executed
tf.debugging.set_log_device_placement(True)

# Define simple tensors and perform matrix multiplication
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print("Result of matrix multiplication:")
print(c)
