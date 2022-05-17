import cv2 as cv
import cv2


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
    inrange = cv.inRange(img, (0, 75, 0), (180, 255, 255))

    cv.createTrackbar('BL', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('GL', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('RL', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('BH', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('GH', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('RH', 'inrange', 0, 255, on_trackbar)
    trackbar_value = 0
    while 1:
        # cv2.imshow('thresh', thresh)
        cv2.imshow('inrange', inrange)
        cv2.imshow('image', img)
        BL = cv2.getTrackbarPos('BL', 'inrange')
        GL = cv2.getTrackbarPos('GL', 'inrange')
        RL = cv2.getTrackbarPos('RL', 'inrange')
        BH = cv2.getTrackbarPos('BH', 'inrange')
        GH = cv2.getTrackbarPos('GH', 'inrange')
        RH = cv2.getTrackbarPos('RH', 'inrange')
        low_values = (BL, GL, RL)
        high_values = (BH, GH, RH)
        print(low_values,',', high_values)
        inrange = cv.inRange(img, low_values, high_values)

        # cv2.imshow('gray', gray)
        # trackbar_value = cv2.getTrackbarPos('123', 'thresh')
        # print(trackbar_value)
        # thresh = cv2.threshold(gray, 0, trackbar_value, cv2.THRESH_OTSU)[1]
        cv2.waitKey(10)
