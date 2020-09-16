from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[3] = pts[np.argmin(s)]
	rect[0] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[2] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights
    Args:
        corners: list
    Returns:
        destination_corners: list
        height: int
        width: int
    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    
    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)
        
    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w
def unwarp(img, src, dst):
    """
    Args:
        img: np.array
        src: list
        dst: list
    Returns:
        un_warped: np.array
    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    return un_warped


def find_rectangle(image_path):
    image = cv2.imread(image_path)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
   
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ## filter label white color
    kernel = np.ones((5,5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    ## threshold the label
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)

    ## find the contours
    canvas = np.zeros(image.shape, np.uint8)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    approx_corners = np.array(approx_corners, dtype='float32')
    # print(np.zeros((4, 2), dtype = "float32"))
    approx_corners = order_points(approx_corners)
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    
    destination_corners, h, w = get_destination_points(approx_corners)
    print(destination_corners)

    perspective_fix =   unwarp(image, np.array(approx_corners), destination_corners)
    cropped = perspective_fix[0:h, 0:w]
    cropped = cv2.flip(cropped, -1)
    cv2.imshow('Frame', thresh)
    k = cv2.waitKey(0)

    if k == 27:         # If escape was pressed exit
        cv2.destroyAllWindows(image)

find_rectangle('C:/Users/Egor_Pilat/Desktop/spot_label/SportLabels/imagenet/data/val/size/284.jpg')


