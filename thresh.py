import cv2 as cv
import cv2

yellow_lower, yellow_upper = (42, 94, 139), (101, 149, 186)


def resize_image(image, scale):
    scale = scale
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    img = cv2.imread('train/img_012.jpg')
    img = resize_image(img, 40)
    img = cv.medianBlur(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    # th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    # th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv.THRESH_BINARY,11,2)
    edges = cv.Canny(gray, 0, 100)
    thresh = cv.inRange(img, yellow_lower, yellow_upper)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours:" + str(len(contours)))
    corners_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 25 and h > 25:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

        corners_list.append((x, y, x+w, y+h))
        


    while 1:
        cv2.imshow('th', thresh)
        cv2.imshow('edges', edges)
        # cv2.imshow('1', th2)
        # cv2.imshow('2', th3)
        cv2.imshow('imdasage', gray)
        cv2.imshow('image', img)
        cv2.waitKey(10)
