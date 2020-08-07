import numpy as np
import cv2
import time
from skimage.util import view_as_windows,pad
from functools import reduce


def bin2int(binary_list):
    """
    Convert binary array list to integer (decimal)
    :param binary_list: list of binary integer i.e. [1,0,1]
    :return: decimal equivalent value of binary numbers i.e. 5 for [1,0,1]
    """
    return binary_list.dot(1 << np.arange(binary_list.shape[-1] - 1, -1, -1))


def derivate_image(im, angle):
    '''
    Compute derivative of input image
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019 
    https://github.com/AdrianUng/palmprint-feature-extraction-techniques
    Modified by: Omar Hassan - August 2020
    ###################################################################
    :param im: input image. should be grayscale!
    :param angle: 0 or 90 degrees
    :return: computed derivative along that direction.
    includes padding...
    '''
    h, w = np.shape(im)
    pad_im = np.zeros_like(im)
    if angle == 'horizontal':  # horizontal derivative
        pad_im[:,:w-1] = im[:,1:]
        pad_im[:,-1] = im[:,-1]
        deriv_im = pad_im - im  # [1:, :w]
    elif angle == 'vertical':
        pad_im[1:,:] = im[:h-1,:]
        pad_im[0,:] = im[0,:]
        deriv_im = pad_im - im  # [1:, :w]
    return deriv_im

###################################################################
def ltrp_first_order(im_d_x, im_d_y):
    """
    Extract LTrP1 code (4 orientations) by using input dx and dy matrices.
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019 
    https://github.com/AdrianUng/palmprint-feature-extraction-techniques
    ###################################################################
    :param im_d_x: derivative of image according to x axis (horizontal)
    :param im_d_y: derivative of image according to y axis (vertical)
    :return: encoded LTrP1 code. Possible values ={1,2,3,4}
    """
    encoded_image = np.zeros(np.shape(im_d_y))  # define empty matrix, of the same shape as the image...

    # # apply conditions for each orientation...
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y >= 0)] = 1
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y >= 0)] = 2
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y < 0)] = 3
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y < 0)] = 4

    return encoded_image

def ltrp_second_order_slow(ltrp1_code):
    """
    Extracting the P-components for every pixel (g_c), as defined in the original paper by S. Murala, R. P. Maheshwari
    and R. Balasubramanian (2012), "Local Tetra Patterns: A New Feature Descriptor for Content-Based Image Retrieval,"
    in IEEE Transactions on Image Processing, vol. 21, no. 5, pp. 2874-2886, May 2012. doi: 10.1109/TIP.2012.2188809.
    This implementation does not consider the MAGNITUDE, but that feature can be easily implemented...
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019
    https://github.com/AdrianUng/palmprint-feature-extraction-techniques
    ###################################################################
    :param ltrp1_code: previously computed LTrP1 code (with 4 possible orientations)
    :return: the P-components stacked together. Output shape = (12, image_size, image_size)
    """
    im_side = np.shape(ltrp1_code)[0]
    ltrp1_code = np.pad(ltrp1_code, (1, 1), 'constant', constant_values=0)
    g_c1 = np.zeros((3, im_side, im_side))
    g_c2 = np.zeros((3, im_side, im_side))
    g_c3 = np.zeros((3, im_side, im_side))
    g_c4 = np.zeros((3, im_side, im_side))

    t = time.time()
    height,width = ltrp1_code.shape[:2]

    iter_ = 0
    for i in range(1, im_side+1):
        for j in range(1, im_side+1):
            iter_ += 1
            g_c = ltrp1_code[i, j]

            # # extract neighborhood around g_c pixel
            neighborhood = np.array([ltrp1_code[i + 1, j], ltrp1_code[i + 1, j - 1], ltrp1_code[i, j - 1],
                                     ltrp1_code[i - 1, j - 1], ltrp1_code[i - 1, j], ltrp1_code[i - 1, j + 1],
                                     ltrp1_code[i, j + 1], ltrp1_code[i + 1, j + 1]])


            # # determine the codes that are different from g_c
            mask = neighborhood != g_c
            # # apply mask
            ltrp2_local = np.multiply(neighborhood, mask)

            # # construct P-components for every orientation.
            if g_c == 1:
                for direction_index, direction in enumerate([2, 3, 4]):
                    g_dir = ltrp2_local == direction
                    r = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))
                    g_c1[direction_index, i - 1, j - 1] = r

            elif g_c == 2:
                for direction_index, direction in enumerate([1, 3, 4]):
                    g_dir = ltrp2_local == direction
                    g_c2[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))

            elif g_c == 3:
                for direction_index, direction in enumerate([1, 2, 4]):
                    g_dir = ltrp2_local == direction
                    g_c3[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))

            elif g_c == 4:
                for direction_index, direction in enumerate([1, 2, 3]):
                    g_dir = ltrp2_local == direction

                    g_c4[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))
                    pass

            elif g_c not in [1, 2, 3, 4]:
                raise Exception('Error - Invalid value for g_c. List of possible values include [1,2,3,4].')

    # # collect all P-components in a 'large_g_c'
    large_g_c = []
    for this_g_c in [g_c1, g_c2, g_c3, g_c4]:
        large_g_c.extend(this_g_c)
    large_g_c = np.array(large_g_c)

    return large_g_c

def ltrp_second_order_fast(ltrp1):
    """
    Faster implementation to compute second order ltrp code. The code
    only computes tetra patterns and does not compute magnitude pattern
    however the code may be extended to compute magnitude as well following
    similar structure.

    :param ltrp1: first order ltrp code
    :return: second order ltrp code
    #######################################################
    Implemented by Omar Hassan - August 2020
    #######################################################
    """
    #get img dimension, currently only supports square dim
    im_side = ltrp1.shape[0]

    #pad input with zeros
    ltrp1 = np.pad(ltrp1, (1, 1), 'constant', constant_values=0)
    #convert to type int to save memory used in next step
    ltrp1 = ltrp1.astype(np.int)
    #generate patches that will be used for performing ltrp steps
    patches = view_as_windows(ltrp1,window_shape=(3,3),step=1)
    #reshape patches to 3x3 shape, same as kernel size
    patches = patches.reshape(-1,3,3)

    #center pixels
    g_c = patches.copy()[:,1,1]
    g_c = g_c.astype(np.uint8)
    g_c = g_c.reshape(-1,1)

    #set center pixels = -1 to later filter and remove from neighbor array
    patches[:,1,1] = -1

    #reshape to vector
    neighbor = patches.reshape(patches.shape[0],-1)

    #retain original number of patches
    patch_num = neighbor.shape[0]

    #reshape to perform boolwise element selection
    neighbor = neighbor.reshape(-1,)
    #filter out array values if they equal to -1
    nmask = neighbor==-1
    neighbor = neighbor[~nmask]
    #reshape back to original number of patches
    neighbor = neighbor.reshape(patch_num,-1)
    #convert to uint8
    neighbor = neighbor.astype(np.uint8)

    #if central pixel equals neighbor in every patches, set them to 0
    mask = neighbor != g_c
    neighbor = np.multiply(neighbor,mask)

    #for ltrp, we generate feature maps for 4(tetra) directions,
    #for each direction, we exclude its value from the patches
    #similarly, we also exclude center pixel of patch
    directions = np.array([[2, 3, 4],
                            [1, 3, 4],
                            [1, 2, 4],
                            [1, 2, 3]])

    #our original patch are indexed as [0,1,2,3,4,5,6,7] (g_c and current_direction pixels are removed from patch)
    #where 0 is top-left, and 7 is bottom right
    #in algorithm, indexed are done as [3,2,1,4,0,5,6,7]
    #where 0 is mid-right, and 7 is bottom right
    #hence we need to swap our indexes

    # swap_idx = np.array([3,2,1,4,0,5,6,7]) # Original Author uses these indexes

    swap_idx = np.array([6,5,3,0,1,2,4,7])  # Referenced Implementation uses these indexes

    neighbor = neighbor[:,swap_idx]

    g_c1234 = []

    for i in range(4):
        #get an array where only central pixel == 1 are there, rest are zero
        #we do this for doing computations related to centeral_pixel==1
        gc_i = g_c.copy()
        #since loop starts from 0, mask index is i+1 here.
        #(i+1) actually means current direction
        #for each direction, we will compute feature maps
        gc_i[gc_i!=(i+1)] = 0
        #reshape to 1d-array
        gc_i = gc_i.reshape(-1,)

        temp = neighbor.copy()

        masks = [temp[gc_i.astype(np.bool)] == x for x in directions[i]]

        pattern = [bin2int(x) for x in masks]

        pattern = np.array(pattern)

        g_c_img = np.zeros((3,im_side*im_side),dtype=np.uint8)

        g_c_img[:,gc_i.astype(np.bool)] = pattern

        g_c1234.append(g_c_img)

    g_c1234 = np.array(g_c1234)

    g_c1234 = g_c1234.reshape(-1,im_side,im_side)

    return g_c1234

def get_ltrp(image,original=False):
    """
    Computes ltrp as a whole (both first and second order).

    :param image: input grayscale image
    :param original: whether to use Adrian's implementation or mine
    :return: 12 ltrp feature maps
    """

    image = np.array(image, dtype=np.float32)  # /255.

    deriv_h = derivate_image(image, 'horizontal')
    deriv_v = derivate_image(image, 'vertical')


    #computer first order ltrp
    ltrp1 = ltrp_first_order(im_d_x=deriv_h, im_d_y=deriv_v)

    #compute second order ltrp
    if(original):
        ltrp2 = ltrp_second_order_slow(ltrp1)
    else:
        ltrp2 = ltrp_second_order_fast(ltrp1)

    return ltrp2

def concat_fm(fm):
    """
    Concatenates Directional feature maps as shown in original paper.
    This function is used for visualization purposes only.

    :param fm: 12 ltrp feature maps
    :return: list of 4 concatenated feature maps
    """

    d1 = fm[0]+fm[1]+fm[2]
    d2 = fm[3]+fm[4]+fm[5]
    d3 = fm[6]+fm[7]+fm[8]
    d4 = fm[9]+fm[10]+fm[11]

    return [d1,d2,d3,d4]

if __name__ == '__main__':
    img = cv2.imread('images/input.png',0)

    t = time.time()
    sfm = get_ltrp(img,original=True)
    print('Original Implementation Time (ms): ',round((time.time() - t)*1000,2))

    t = time.time()
    ffm = get_ltrp(img)
    print('Faster Implementation Time (ms): ',round((time.time() - t)*1000,2))

    print('Difference between Implementations: ',np.sum(np.abs(sfm-ffm)))

    #write feature maps to drive
    fm_list = concat_fm(ffm)
    for i,fm in enumerate(fm_list):
        cv2.imwrite(f"images/output{i}.jpg",fm)
