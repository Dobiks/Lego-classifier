import cv2 as cv
import cv2

red_lower, red_upper =  (0, 0, 0), (70, 75, 255)
blue_lower, blue_upper = (111, 0, 0), (255, 116, 115)
green_lower, green_upper = (0, 0, 0), (104, 128, 67)
yellow_lower, yellow_upper =  (42, 94, 139), (101, 149, 186)

def resize_image(image, scale):
    scale = scale
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def on_trackbar(val):
    return


if __name__ == '__main__':
    img = cv2.imread('train/img_011.jpg')
    img = resize_image(img, 40)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('inrange')
    thresh = cv2.threshold(gray, 255, 255, cv2.THRESH_OTSU)[1]
    inrange = cv.inRange(img, blue_lower, blue_upper)
    inrange += cv.inRange(img, green_lower, green_upper)
    inrange += cv.inRange(img, red_lower, red_upper)
    inrange += cv.inRange(img, yellow_lower, yellow_upper)

    contours,hierarchy = cv2.findContours(inrange,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours:" + str(len(contours)))
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w > 40 and h > 40:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    while 1:
        # cv2.imshow('thresh', thresh)
        cv2.imshow('inrange', inrange)
        cv2.imshow('image', img)
        cv2.waitKey(10)
