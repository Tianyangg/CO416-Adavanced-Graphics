import sys
import numpy as np
from PNM import *

def CreateAndSavePFM(out_path):
    width = 512
    height = 512
    numComponents = 3

    img_out = np.empty(shape=(width, height, numComponents), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = 1.0

    writePFM(out_path, img)

def LoadAndSavePPM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePPM(out_path, img_out)

def LoadAndSavePFM(in_path, out_path):
    img_in = loadPFM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePFM(out_path, img_out)

def LoadPPMAndSavePFM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:]/255.0

    writePFM(out_path, img_out)

def LoadPFMAndSavePPM(in_path, out_path):
    img_in = loadPFM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] * 255.0

    writePPM(out_path, img_out.astype(np.uint8))

def get_Norm(row, col, r):
    """
    change the Origin from top left to middld of sphere, calculate the corresponding z
    :param row: row index of the image
    :param col: column index of the image
    :param r: radius of the image (in this case should be 511/2)
    :return: normalized vecotor [x y z]
    """
    x = float(col - r)  # shift the coordinates
    y = float(r - row)
    z = np.sqrt(r ** 2 - x ** 2 - y ** 2)/r
    return np.array([x/r, y/r, z])

def xyz2sphere(vec, h, w):
    """
    change from cartesian cordinates to spherical coordinates.
    :param vector: normalized vector
    :return: phi, theta
    """
    x, y, z = vec[0], vec[1], vec[2]
    # if y <= -1:
    #     print x, y, z
    phi = np.arctan2(x, z)
    # theta = np.arccos(y/np.sqrt(x ** 2 + y ** 2 + z ** 2))
    theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), y)
    phi += np.pi  # shift from [-pi, pi] to [0, 2pi]
    # print phi
    # print "theta: ", theta, "y: ", y

    return int(phi/(2*np.pi)*w), int(theta/np.pi*h)

def gamma_correction(input, gamma, stop, output):
    """
    Perform gamma gamma_correction
    """
    # load the pmf image (mirror ball one)
    image = loadPFM(input)
    h, w, channel = image.shape

    process_image = np.empty(shape=(h, w, channel), dtype = np.float32)
    process_image = np.multiply(image, stop)

    process_image = np.power(process_image, 1/gamma)

    process_image[process_image > 1] = 1
    process_image = np.ceil(np.multiply(process_image, 255))

    writePPM(output, process_image.astype(np.uint8))


def generate_map(input, out):
    """
    load a lat long file, pixel index it to another file with 511*511
    """
    # load latlong file
    old_img = loadPFM(input)
    height, width, channel = old_img.shape

    # define the empty image for mapping
    d = 511
    radius = 511./2
    view_vec = [0.0, 0.0, 1.0]   # make sure this is float32
    new_Image = np.empty(shape=(d, d, channel), dtype=np.float32)

    #pixel indexing
    for i in range(0, d):
        for j in range(0, d):
            if (radius - i)**2 + (radius - j)**2 < radius**2:
                norm = get_Norm(i, j, radius)
                # calculate reflection
                ref_vec = np.dot(norm, view_vec) * 2 * norm - view_vec
                x, y = xyz2sphere(ref_vec, height, width)

                for c in range(channel):
                    new_Image[i][j][c] = old_img[y][x][c]
                    if new_Image[i][j][c] > 1:
                        new_Image[i][j][c] = 1.0

    writePFM(out, new_Image)



if '__main__' == __name__:
    # LoadAndSavePFM('grace_latlong.pfm', 'test.pfm')
    # LoadAndSavePPM('9.ppm', 'test.ppm')
    # LoadPFMAndSavePPM('test.pfm', 'grace.ppm')
    # LoadPPMAndSavePFM('test.ppm', '9.pfm')
    # generate_map('/Users/tianyangsun/Documents/Imperial_S2/Advanced_graphics/CO417-Assignment1/UrbanProbe/urbanEM_latlong.pfm', "/Users/tianyangsun/Documents/Imperial_S2/Advanced_graphics/CO417-Assignment1/UrbanProbe/result2.pfm")
    gamma_correction('../UrbanProbe/result2.pfm', 1.4, 3, '../UrbanProbe/result2AfterGamma.pfm' )
