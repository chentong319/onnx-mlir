# Test: no name for input and output
import numpy as np
import onnxmlirdocker

#sess = onnxmlirdocker.InferenceSession("test_add.so")
sess = onnxmlirdocker.InferenceSession("test_add.onnx")

a = np.zeros((3, 4, 5), dtype=np.float32)
b = a + 4

sess.run(None, [a, b])
r = sess.run(None, [a, b])
print(r)
