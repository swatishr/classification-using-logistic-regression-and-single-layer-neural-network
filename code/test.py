
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

test_image = Image.open('../proj3_images/Test/test_0001.png')
new_image = make_square(test_image)
new_image.show()