import numpy as np
import cv2
import sys

# Globals
gaussian_distance_table = None
gaussian_rangeDiff_table = np.zeros(256)
RED = 0
GREEN = 1
BLUE = 2


def euclidean_distance(x1, y1, x2, y2):
    return np.linalg.norm([x1 - x2, y1 - y2])


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return np.exp(-0.5 * np.square(x/sigma))


def range_different(source_matrix, x, y, neigh_x, neigh_y):
    # distance between the colors of the pixels in the matrix
    return int(source_matrix[x, y]) - int(source_matrix[neigh_x, neigh_y])


def apply_filter_grayscale(source_matrix, x, y, sig_domain, sig_range, radius):  #For a specific pixel
    total_weight = 0
    filtered_pixel_i = 0
    i = max(0, x-radius)
    while i <= min(source_matrix.shape[0]-1, x+radius):
        j = max(0, y - radius)
        while j <= min(source_matrix.shape[1]-1, y+radius):
            neigh_x = i
            neigh_y = j
            gauss_distance = gaussian_distance_table[abs(x - neigh_x)][abs(y - neigh_y)]
            gauss_sim = gaussian_rangeDiff_table[abs(int(source_matrix[x][y]) - int(source_matrix[neigh_x][neigh_y]))]
            local_weight = gauss_distance * gauss_sim
            filtered_pixel_i += source_matrix[neigh_x][neigh_y] * local_weight
            total_weight += local_weight
            j += 1
        i += 1
    filtered_pixel_i = filtered_pixel_i / total_weight
    return int(round(filtered_pixel_i))


def apply_filter_color(source_matrix, x, y, sig_domain, sig_range, radius, color):  #For a specific pixel
    total_weight = 0
    filtered_pixel_i = 0
    i = max(0, x-radius)
    while i <= min(source_matrix.shape[0]-1, x+radius):
        j = max(0, y - radius)
        while j <= min(source_matrix.shape[1]-1, y+radius):
            neigh_x = i
            neigh_y = j
            gauss_distance = gaussian_distance_table[abs(x - neigh_x)][abs(y - neigh_y)]
            gauss_sim = gaussian_rangeDiff_table[abs(int(source_matrix[x][y][color]) - int(source_matrix[neigh_x][neigh_y][color]))]
            local_weight = gauss_distance * gauss_sim
            filtered_pixel_i += source_matrix[neigh_x][neigh_y][color] * local_weight
            total_weight += local_weight
            j += 1
        i += 1
    filtered_pixel_i = filtered_pixel_i / total_weight
    return int(round(filtered_pixel_i))


def filter_matrix_grayscale(source_matrix, sig_dom, sig_range, rad):
    filtered_matrix = np.zeros(source_matrix.shape)
    for i in range(filtered_matrix.shape[0]):
        print("Row #" + str(i) + " out of " + str(filtered_matrix.shape[0]))
        for j in range(filtered_matrix.shape[1]):
            filtered_matrix[i, j] = apply_filter_grayscale(source_matrix, i, j, sig_dom, sig_range, rad)
    return filtered_matrix


def filter_matrix_color(source_matrix, sig_dom, sig_range, rad):
    filtered_matrix = np.zeros(source_matrix.shape)

    for color in range(3):
        for i in range(filtered_matrix.shape[0]):
            print("Row #" + str(i) + " out of " + str(filtered_matrix.shape[0]))
            for j in range(filtered_matrix.shape[1]):
                filtered_matrix[i][j][color] = apply_filter_color(source_matrix, i, j, sig_dom, sig_range, rad, color)

    return filtered_matrix



def init_gaussian_tables(radius, sig_domain, sig_range):
    """
    This function initiates the values for the gaussian distances & range differences.
    :param radius: how many pixels to check in each direction
    :param sig_domain: the squared root of the variance of the domain
    :param sig_range: the squared root of the  variance of the range
    :return: void
    """
    size = radius+1
    global gaussian_distance_table
    global gaussian_rangeDiff_table
    gaussian_distance_table = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            dist = np.linalg.norm([i, j])
            gaussian_distance_table[i][j] = gaussian(dist, sig_domain)
    for i in range(256):
        gaussian_rangeDiff_table[i] = gaussian(i, sig_range)



def main_for_color():
    print('Input image: ' + str(sys.argv[1]))
    radius = int(input('Insert radius:\n'))
    sig_domain = int(input('Insert domain variance:\n'))
    sig_range = int(input('Insert range variance:\n'))
    init_gaussian_tables(radius, sig_domain, sig_range)
    src = str(sys.argv[1])
    source_matrix = cv2.imread(src, cv2.IMREAD_COLOR)
    output_matrix = filter_matrix_color(source_matrix, sig_domain, sig_range, radius)
    pic = cv2.imwrite(str(sys.argv[2]), output_matrix)
    cv2.waitKey()
    return

def main_for_grayscale():
    src = str(sys.argv[1])
    mat = cv2.imread(src, cv2.IMREAD_COLOR)
    print('Input image: ' + str(sys.argv[1]))
    radius = int(input('Insert radius:\n'))
    sig_domain = int(input('Insert domain variance:\n'))
    sig_range = int(input('Insert range variance:\n'))
    init_gaussian_tables(radius, sig_domain, sig_range)
    src = str(sys.argv[1])
    source_matrix = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    output_matrix = filter_matrix_grayscale(source_matrix, sig_domain, sig_range, radius)
    pic = cv2.imwrite(str(sys.argv[2]), output_matrix)
    cv2.waitKey()
    return


if __name__ == "__main__":
    if str(sys.argv[3]) == "color":
        main_for_color()
    else:
        main_for_grayscale()
