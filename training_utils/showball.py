import matplotlib.pyplot as plt
import os

def get_grayscale_fovea(file):
    fovea_grey = []
    with open(file) as f:
      for line in f:
        fovea_grey.append([int(el) for el in line.split()])
    f.close()
    return fovea_grey


def display(fovea):
    plt.imshow(fovea,  interpolation='nearest')
    plt.show()


path = "training/"
for file in os.listdir(path):
    if not file.endswith(".png"):
        print(file)
        gray_fovea = get_grayscale_fovea(path + file)
        display(gray_fovea)


