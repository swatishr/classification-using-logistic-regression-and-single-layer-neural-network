
#Library file to extract USPS test data

import os
import numpy as np
import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing

#Function to shape the image such that height = width
def make_square(im):
    x, y = im.size
    fill_color=(255, 255, 255, 0)
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    a = int((size - x) / 2)
    b = int((size - y) / 2)
    new_im.paste(im, (a, b))
    return new_im

#Function to get USPS test images and labels
def extract_usps_data(scale):
	try:
		data = np.load("USPStestData.npz")
		usps_test_images = data['usps_test_images']
		usps_test_labels = data['usps_test_labels']
	except FileNotFoundError:
		listImages = []
		usps_test_images = []
		usps_test_labels = np.zeros([19999, 10], dtype=np.int)
	
		j = 0
		for i in range(10):
			for file in os.listdir("../proj3_images/Numerals/"+str(i)+"/"):
				if file.endswith(".png"):
					
					test_image = Image.open("../proj3_images/Numerals/"+str(i)+"/"+file)
					new_image = make_square(test_image)
					image = new_image.convert('L')
					image = PIL.ImageOps.invert(image)
					image = image.resize((28, 28), Image.BICUBIC)
					img_array = np.asarray(image)
					if scale!=1:
						img_normalized = (img_array - img_array.min())/(img_array.max() - img_array.min())
						usps_test_images.append(img_normalized.flatten())
					else:
						usps_test_images.append(img_array.flatten())
					usps_test_labels[j, i] = 1
					j = j+1

		usps_test_images = np.asarray(usps_test_images)
		# np.savez("USPStestData.npz", usps_test_images=usps_test_images, usps_test_labels=usps_test_labels)
	
	return usps_test_images, usps_test_labels



# usps_test_images, usps_test_labels = extract_usps_data(0)
# print(usps_test_images.shape, usps_test_labels.shape)