from skimage import io, color, util

img = io.imread("kodim21.png")          # color image: H x W x 3 bytes
gray = util.img_as_ubyte(color.rgb2gray(img))   # grayscale: H x W bytes
io.imsave("kodim21.pgm", gray)
print(gray.shape)