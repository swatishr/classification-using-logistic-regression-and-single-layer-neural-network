
#To extract USPS test data

import os
import numpy as np
import PIL.ImageOps
from PIL import Image

def make_square(im):
    x, y = im.size
    print(im.size)
    fill_color=(255, 255, 255, 0)
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    a = int((size - x) / 2)
    b = int((size - y) / 2)
    #new_im.paste((a/2, b/2), im)
    new_im.paste(im, (a, b))
    return new_im

def usps_data():
	test_image = Image.open('../proj3_images/Test/test_0001.png')
	new_image = make_square(test_image)
	image = new_image.convert('L')
	image = PIL.ImageOps.invert(image)
	return image


image = usps_data()
image = image.resize((28, 28), Image.BICUBIC)
print(image.size)
image.show()
img_array = np.asarray(image)
img_normalized = (img_array - img_array.min())/(img_array.max() - img_array.min())
print(img_normalized)
img = Image.fromarray(img_normalized)
img.show()

# usps_test_images

# for file in os.listdir("../proj3_images/Test/"):
# 	if file.endswith(".png"):
# 		#listImages.append(file)

# image = Image.open('../proj3_images/Test/test_0001.png').convert('L')
# inv_image = PIL.ImageOps.invert(image)

# img_resize = image.resize((28, 28), Image.ANTIALIAS)

#read image in grayscale
# image = cv2.imread('../proj3_images/Test/test_0001.png', 0)
# cv2.imshow('hello usps',image)
# cv2.waitKey(0)
# img_array = np.asarray(image)
# # img_normalized = (img_array - img_array.min())/(img_array.max() - img_array.min())
# # #img_array_inv = inv(img_normalized)
# # a = np.absolute(img_normalized-1)
# print(img_array)

# # plt.imshow(img_array)
# # plt.show()

# #USPS Test labels
# usps_test_labels = np.zeros([1500, 10], dtype=np.int)
# # usps_test_labels[0:150, 9] = 1
# # usps_test_labels[150:300, 8] = 1
# # usps_test_labels[300:450, 7] = 1
# # usps_test_labels[450:600, 6] = 1
# # usps_test_labels[600:750, 5] = 1
# # usps_test_labels[750:900, 4] = 1
# # usps_test_labels[900:1050, 3] = 1
# # usps_test_labels[1050:1200, 2] = 1
# # usps_test_labels[1200:1350, 1] = 1
# # usps_test_labels[1350:1500, 0] = 1

# batch_size = 150
# j = 9
# for i in range(9):
# 	usps_test_labels[i * batch_size : (i+1) * batch_size, j] = 1
# 	j = j-1;

#USPS test images
