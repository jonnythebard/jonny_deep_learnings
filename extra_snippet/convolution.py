import numpy as np

INPUT_HEIGHT = 32
INPUT_WIDTH = 32
KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3

output_height = INPUT_HEIGHT - KERNEL_HEIGHT + 1
output_width = INPUT_WIDTH - KERNEL_WIDTH + 1

input_image = np.random.random((INPUT_HEIGHT, INPUT_WIDTH))
kernel = np.random.random((KERNEL_HEIGHT, KERNEL_WIDTH))
output_image = np.zeros((output_height, output_width))

for i in range(0, output_height):
    for j in range(0, output_width):
        for _i in range(0, KERNEL_HEIGHT):
            for _j in range(0, KERNEL_WIDTH):
                output_image[i, j] += input_image[i-_i, j-_j] * kernel[_i, _j]
