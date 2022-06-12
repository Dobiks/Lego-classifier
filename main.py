import sys
import cv2 as cv
import cv2
import glob
import numpy as np
import random


class LegoDetection:
    def __init__(self) -> None:
        self.files = glob.glob('train/*.jpg')
        self.shapes = glob.glob('shapes/*.png')

    def get_masked_img(self, img_color):

        hsv = cv2.cvtColor(img_color, cv.COLOR_BGR2HSV)
        hls = cv2.cvtColor(img_color, cv.COLOR_BGR2HLS)
        white1 = cv.inRange(hsv, (7, 0, 170), (255, 255, 189))
        colors = cv.inRange(hls, (0, 57, 0), (255, 232, 68))
        green = cv.inRange(hls, (34, 0, 0), (131, 93, 255))
        colors = 255 - colors
        masks = colors + green + white1
        final = cv.bitwise_and(img_color, img_color, mask=masks)
        return masks, final

    def cropp_shapes(self, contours, hierarchy, image):
        shapes = []
        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            if (rect[1][0] > 20 and rect[1][1] > 20) and hierarchy[0][i][3] == -1:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                width = int(rect[1][0])
                height = int(rect[1][1])
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image, M, (width, height))
                if width < height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                shapes.append(warped)
                # cv2.imwrite(f"{len(shapes)}.jpg", warped)
        return shapes

    def match_shape(self, shape):
        best_shape = None
        lowest_score = 666
        for sample_file in self.shapes:
            sample_img = cv2.imread(sample_file, 0)
            #resize shape to match the image
            # size = (100,100)
            # # shape = cv2.resize(shape ,size, interpolation=cv2.INTER_AREA)
            # # shape_img = cv2.resize(shape_img ,size, interpolation=cv2.INTER_AREA)
            # for i in range (0,4):
            #     # shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_CLOCKWISE)
            #     for j in range(0,2):
            #         # shape_img = cv2.flip(shape_img, 0)
            #         for k in range(0,2):
            #             # shape_img = cv2.flip(shape_img, 1)
            contour1, heirarchy = cv2.findContours(sample_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour2, heirarchy = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            score = cv2.matchShapes(contour1[0], contour2[0], cv2.CONTOURS_MATCH_I1, 0)
            # score = cv.matchTemplate(shape,shape_img,cv.TM_CCOEFF_NORMED)


            # gray = cv.cvtColor(shape,cv.COLOR_BGR2GRAY)


            dst = cv.cornerHarris(shape,2,3,0.04)
            bgr = cv.cvtColor(shape,cv.COLOR_GRAY2BGR)
            bgr[dst>0.1*dst.max()]=[0,0,255]
            cv.imshow('dst',bgr)
            cv2.waitKey(0)

            if score < lowest_score and score < 0.1:
                lowest_score = score
                best_shape = sample_img.copy()


        if best_shape is not None:
            print("lowest: ", lowest_score, sample_file)
            cv2.imshow('sample', best_shape)
            cv2.imshow('shape', shape)
            cv2.waitKey(0)

    def detect_corners(self, img):
        operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        operatedImage = np.float32(operatedImage)
        dest = cv2.cornerHarris(operatedImage, 3, 3, 0.22)
        dest = cv2.dilate(dest, None)
        img[dest > 0.01 * dest.max()]=[0, 0, 255]

        cv.imshow('corners', img)
        cv.waitKey(0)

    def detect(self):
        for file in self.files:
            print(file)
            img_color = cv2.imread(file)
            img_color = cv2.resize(img_color, None, fx=0.3, fy=0.3)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_Gauss = cv2.GaussianBlur(img_gray, (5, 5), -1)

            thresh_value = 14
            canny_output = cv2.Canny(img_Gauss, thresh_value, thresh_value * 2)

            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(canny_output, kernel, iterations=2)

            contours, hierarchy = cv2.findContours(
                dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            masks, color_masks = self.get_masked_img(img_color)

            shapes = self.cropp_shapes(contours, hierarchy, img_color)

            for shape in shapes:
                # self.match_shape(shape)
                self.detect_corners(shape)
                # cv2.imshow('org', shape)
                # cv2.waitKey(0)


if __name__ == '__main__':
    LegoDetection().detect()
