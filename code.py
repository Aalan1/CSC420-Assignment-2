import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import data
from skimage import io
from skimage import filters
from scipy import ndimage
from skimage.feature import match_template

def seam_carving(image):
    """
    perform seam carving algorithm on the given image
    """
    # Read image with RGB norm, as discussed on Discussion
    img = io.imread(image)

    new_img = np.zeros((img.shape[0],img.shape[1]))
    
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            # using norm of RGB vector 
            new_img[row][col] = np.linalg.norm(img[row][col])
            
    # (a) Compute Magnitude of gradients image, using RGB norms
    img_x = ndimage.sobel(new_img,axis=0)

    img_y = ndimage.sobel(new_img,axis=1)

    magnitudes = np.zeros(new_img.shape)

    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            # compute magnitude of gradients for each pixel
            magnitudes[row,col] = math.sqrt((img_x[row,col] ** 2) + (img_y[row,col] **2))

    # (b) Find connected path of pixels that has the smallest sum of gradients
    # helper function returns matrix of objects Pixel(valid, current_value, parent_neighbor_index)
    print("Finding paths")
    img_paths = paths(magnitudes)
    final_img = img.copy()
    """
    # extract the lowest path
    first_path = img_paths[0]
    # (c) remove pixels from the lowest path

    for pixel in first_path:
        final_img[pixel[0],pixel[1]] = 0
    plt.figure(1)
    plt.axis("off")
    plt.imshow(final_img)
    plt.show()
    """
    pixels = []
    # (d) Remove more paths
    print("final Matrix")
    for lst in range(len(img_paths)):
        pixels = pixels + img_paths[lst] 
    for pixel in pixels:
        final_img[pixel[0],pixel[1]] = 0
    plt.figure(1)
    plt.axis("off")
    plt.imshow(final_img)
    plt.show()
    

def paths(magnitudes):
    """
    returns a list of all valid paths starting for top to bottom
    using dynamic programming, Valid paths need to have neighbors from
    image.
    """
    final_paths = []

    matrix = np.empty(magnitudes.shape, dtype=object)

    # fill matrix with Pixel objects containg information (valid, value,parent_neighbor)
    # Pixel class is below, containing information from the path algorithm
    # Valid means if a Pixels are marked and cannot be used again
    # value is the pixel magnitude and parent neighbor is the pixels parent in the path

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == 0:
                # current sum is the original magnitude and no parent neighbor
                matrix[row][col] = Pixel(magnitudes[row][col],-10)
            else:
                if col == 0:
                    # left boundary of the image, compare top and top right only
                    lst = [matrix[row-1][col].val,matrix[row-1][col+1].val]
                    matrix[row][col] = Pixel(magnitudes[row][col] + min(lst),lst.index(min(lst)))
                elif col == matrix.shape[1] - 1:
                    # right boundary of image, compare top and top left only
                    lst = [matrix[row-1][col-1].val,matrix[row-1][col].val]
                    matrix[row][col] = Pixel(magnitudes[row][col] + min(lst),lst.index(min(lst))-1)
                else:
                    # compare all three possible paths, pick the smallest
                    lst = [matrix[row-1][col-1].val,matrix[row-1][col].val,matrix[row-1][col+1].val]  
                    matrix[row][col] = Pixel(magnitudes[row][col] + min(lst),lst.index(min(lst))-1)
                
    # now find the paths, first check to see if the lowest total is valid, if not move on to the next lowest and so on
    # get last row indices based on lowest sum
    lowest = [tup.val for tup in matrix[matrix.shape[0] - 1::,::][0]]
    # get the index list sorted
    lowest = sorted(range(len(lowest)), key=lowest.__getitem__)
    
    for index in lowest:
        # check to see if the path is valid
        valid = 0
        current_col = index
        for row in range(matrix.shape[0] - 1, -1, -1):
            # if any pixel in the path is 1, then it is not valid go to next lowest                
            if matrix[row][current_col].is_taken():
                valid = 1    
            current_col = current_col + matrix[row][current_col].neighbor
        current_col = index     
        if valid == 0:
            # the Path is valid and will be picked!
            # add path to the list and mark pixels so they will not be picked again
            valid_path = []
            current_col = index
            for row in range(matrix.shape[0] - 1, -1, -1):
                # traverse from down to up    
                valid_path.append((row,current_col))
                # taken to 1
                matrix[row][current_col].now_taken()
                current_col = current_col + matrix[row][current_col].neighbor
            final_paths.append(valid_path)
    return final_paths

example = np.array([[2,3,4,6,7,8],[1,5,2,2,3,4],[10,2,7,6,8,10],[1,2,2,2,2,3]])

example2 = np.array([[1,4,3,5,2],[3,2,5,2,3],[5,2,4,2,1]])

class Pixel:
    """A pixel class containing
    information on pixel validation for paths, parent neighbor for path,and magnitude intensity
    """

    def __init__(self, value, neighbor):
        self.val = value
        self.taken = 0
        self.neighbor = neighbor

    def now_taken(self):
        self.taken = 1

    def is_taken(self):
        return self.taken == 1

def upscalling(image):
    """
    Given an image, up scale 3 times its size. 
    """
    # read RGB image, then get compute norm of RGBs
    img = io.imread(image)
    new_img = np.zeros((img.shape[0],img.shape[1]))
    
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            # using norm of RGB vector 
            new_img[row][col] = np.linalg.norm(img[row][col])
            
    bigger = np.zeros((new_img.shape[0] * 3,new_img.shape[1] * 3))

    # copy values in every third pixel
    for row in range(bigger.shape[0]):
        for col in range(bigger.shape[1]):
            if (row % 3 == 0) and (col % 3 == 0):
                # copy here
                bigger[row,col] = new_img[row // 3, col // 3]

    # now perform bilinear interpolation
    for row in range(bigger.shape[0]):
        for col in range(bigger.shape[1]):
            if not((row % 3 == 0) and (col % 3 == 0)):   
                # find x1, x2, y1, y2
                x1 = 3 * math.floor(col/3)
                x2 = x1 + 3
                y1 = 3 * math.ceil(row/3)
                y2 = y1 - 3
                x = col
                y = row
                # apply formula
                total = 0
                if (x1 < bigger.shape[1] and y1 < bigger.shape[0]):
                    total += bigger[y1,x1] * abs(x2 - x) * abs(y2 - y)
                if (x1 < bigger.shape[1] and y2 < bigger.shape[0]):
                    total += bigger[y2,x1] * abs(x2 - x) * abs(y - y1)
                if (x2 < bigger.shape[1] and y1 < bigger.shape[0]):
                    total += bigger[y1,x2] * abs(x1 - x) * abs(y2 - y)
                if (x2 < bigger.shape[1] and y2 < bigger.shape[0]):
                    total += bigger[y2,x2] * abs(x - x1) * abs(y - y1)
                bigger[row,col] = round(1/9 * total)    
    return bigger

def Corner_Harris(image, gauss_sigma, threshold):
    """
    Given an image, perform the Corner Harris Detector, place circles on
    the image where the corners are
    """
    #Read the image
    img = io.imread(image,as_gray=True)
    
    
    # (1) Compute gradients Ix and Iy
    Ix = ndimage.sobel(img,axis=0)
    Iy = ndimage.sobel(img,axis=1)
    # (2) Compute Ixx, Iyy, Ixy
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    # (3) Average Gaussian gives M
    g_Ixx = ndimage.gaussian_filter(Ixx, gauss_sigma)
    g_Iyy = ndimage.gaussian_filter(Iyy, gauss_sigma)
    g_Ixy = ndimage.gaussian_filter(Ixy, gauss_sigma)
    
    # (4) Compute R = det(M) - 0.04 * trace(M)^2 for each window image
    final_img = np.zeros(img.shape)
    for row in range(final_img.shape[0]):
        for col in range(final_img.shape[1]):
            first = g_Ixx[row][col]
            second = third = g_Ixy[row][col]
            last = g_Iyy[row][col]
            M = np.array([[first,second],[third, last]])
            determinant = np.linalg.det(M)
            trace = np.matrix.trace(M)
            final_img[row][col] = determinant - (0.04 * trace ** 2)
    # (5) Find points with large R > threshold, print the corners
    plt.figure(1)
    plt.axis("off")
    plt.imshow(img,cmap='gray')
    for row in range(final_img.shape[0]):
        for col in range(final_img.shape[1]):
            if final_img[row][col] > threshold:
                plt.plot(col, row, 'o', markeredgecolor='r', markerfacecolor='none', markersize=1)
    plt.show()
    
