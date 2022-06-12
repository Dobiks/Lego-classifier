from cgitb import grey
import cv2


def on_trackbar(val):
    # edged = cv2.Canny(gray, val, 255)  # Determine edges of objects in an image
    ret, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    cv2.imshow('test', thresh)


if __name__ == '__main__':
    title_window = 'test'
    img = cv2.imread('img_007.jpg')
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
    edged = cv2.Canny(gray, 100, 255)  # Determine edges of objects in an image
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.namedWindow(title_window)

    # cv2.createTrackbar('test', title_window , 0, 254, on_trackbar)
    # cv2.imshow('polygons_d2e41tected', img)
    # cv2.imshow('polygons_de2te24cted', gray)
    cv2.imshow('edged', edged)
    # cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
