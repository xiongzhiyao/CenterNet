import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np

#print(torch.cuda.get_device_name(0))
print("Hello World，with plotting world")

a = np.random.random((100,100))
print(a)
plt.imshow(a)
plt.show()

print("now try something for the source tree")
