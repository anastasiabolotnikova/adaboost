import numpy
import cv2
import os

def get_grayscale_fovea(file):
    return numpy.loadtxt(file)

def createfile(img, file):
	print("Processing " + file)
	print img

	scale = numpy.amax(img)
	for i in range(len(img)):
		for j in range(len(img[i])):
			img[i][j] = 255*(img[i][j] / scale)

	res = cv2.resize(numpy.array(img),None,fx=7, fy=7, interpolation = cv2.INTER_NEAREST)
	cv2.imwrite("../code_ccc/training+/" + str(file) + '_resized.jpg', numpy.array(res))
	#cv2.imwrite("../filterOutNeg/" + str(file) + '.jpg', numpy.array(img))


path = "../code_ccc/training+/"
for file in os.listdir(path):
    if not file.endswith(".jpg"):
      print(file)
      gray_fovea = get_grayscale_fovea(path + file)
      createfile(gray_fovea, file)
    else:
    	os.remove(path+file)
