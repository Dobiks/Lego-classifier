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
    edges = cv.Canny(gray,100,200)

    cv.createTrackbar('BL', 'inrange', 0, 255, on_trackbar)
    cv.createTrackbar('max', 'inrange', 0, 255, on_trackbar)
    while 1:
        cv2.imshow('inrange', edges)
        cv2.imshow('image', img)
        trk = cv2.getTrackbarPos('BL', 'inrange')
        mx = cv2.getTrackbarPos('max', 'inrange')
        if trk<mx:
            edges = cv.Canny(gray,trk,mx)

        cv2.waitKey(10)
