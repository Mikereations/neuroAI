import numpy as np
import matplotlib.pyplot as plt
import random
import PIL 
import cv2


def getLinePositiveSlope(m,b, size):
    """
    Generates a line with positive slope.
    
    Args:
        m (float): The slope of the line.
        b (float): The y-intercept of the line.
        size (int): The size of the image.
    
    Returns:
        tuple: The line.
    """
    if b >= 0 and b <= 64 : 
        start_y = b
        start_x = 0
    else:
        start_y = 0 
        start_x = -b / m
    
    if (size*m + b) >= 0 and (size*m + b) <= size:
        end_y = size * m + b
        end_x = size
    else :
        end_y = size
        end_x = (size - b) / m
    return start_x, start_y, end_x, end_y

def getLineNegativeSlope(m,b, size):
    """
    Generates a line with negative slope.
    
    Args:
        m (float): The slope of the line.
        b (float): The y-intercept of the line.
        size (int): The size of the image.
    
    Returns:
        tuple: The line.
    """
    if b >= 0 and b <= size : 
        start_y = b
        start_x = 0
    else:
        start_y = size
        start_x = (size - b) / m
    
    if (-b / m) >= 0 and (-b / m) <= size:
        end_y = 0
        end_x = -b / m
    else :
        end_y = size * m + b
        end_x = size
    return start_x, start_y, end_x, end_y

def intersect(line1, line2):
    """
    Finds the intersection of two lines in 2D, whether it happens within the image or not.
    
    Args:
        line1 (tuple): The first line.
        line2 (tuple): The second line.
    
    Returns:
        tuple: The intersection point.
    """
    try :
        intersection_x = (line2[1] - line1[1]) / (line1[0] - line2[0])
        intersection_y = line1[0] * intersection_x + line1[1]
    except ZeroDivisionError : 
        intersection_x = -1
        intersection_y = -1
    return (intersection_x, intersection_y)

# Generate data
def generateData():
    # set the size of the image
    size = 64
    # Create a 2D array of zeros
    flag = True
    while flag:
        data = np.zeros((size, size), dtype=np.uint8)
        lines = []
        # Generate the two random lines 
        for i in range(2):
            # Generate random points
            x1 = random.randint(0, size - 1)
            y1 = random.randint(0, size - 1)
            x2 = x1 
            while x2 == x1 : 
                x2 = random.randint(0, size - 1)
            y2 = y1
            while y2 == y1 : 
                y2 = random.randint(0, size - 1)
            # get the line equation
            m = (y2 - y1) / (x2 - x1)
            lines.append((m, y2 - m*x2))
            # Draw the line
            if lines[-1][0] > 0 :
                start_x, start_y, end_x, end_y = getLinePositiveSlope(lines[-1][0], lines[-1][1], size)
            else:
                start_x, start_y, end_x, end_y = getLineNegativeSlope(lines[-1][0], lines[-1][1], size)
            data = cv2.line(data, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1, 1)
            # Save the data
            plt.imsave('./data.png', data, cmap='gray')
        # Find the intersection point
        intersection = intersect(lines[0], lines[1])
        if (intersection[0] < 0 or intersection[0] > size or intersection[1] < 0 or intersection[1] > size):
            continue
        else:
            flag = False
    return data

def gen_batch(batch_size) : 
    """
    Generates a batch of data.
    
    Args:
        batch_size (int): The size of the batch.
    
    Returns:
        list: The batch of data.
    """
    batch = np.zeros((batch_size, 64, 64), dtype=np.uint8)
    for i in range(batch_size):
        batch[i] = generateData()
        plt.imsave('./inputs/data_{}.png'.format(i), batch[i], cmap='gray')
    return batch
    
if __name__ == '__main__':
    gen_batch(64)