import cv2
import numpy as np


def main():
    # https: // gist.github.com / gyurisc / aa692ad27c2699ec3700

    # Ejemplo de uso

    # Imagen con codigo de barra
    img = "examplebarcode.jpg"

    # Inicializar detector de codigo
    det = BarCodeDetector()

    # Detectar y marcar contorno de codigo
    detected = det.detect_barcode(img, threshold=150)

    # Guardar imagen con contorno detectado
    det.save_img(detected, 'detected.jpg')


class BarCodeDetector:
    def __init__(self):
        pass

    @staticmethod
    def get_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def get_image_grayscale(image_path):
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def save_img(image, save_path):
        cv2.imwrite(save_path, image)

    @classmethod
    def detect_barcode(cls,
                       image_path,
                       threshold=200,
                       contour_color=(0, 255, 0),
                       rgb=False):

        img = cls.get_image(image_path)
        gray_img = cls.get_image_grayscale(image_path)
        gradient_image = cls.get_image_gradient(gray_img)
        threshold_image = cls.get_threshold_image(gradient_image, threshold)
        dilated_image = cls.kernel_erode_dilate(threshold_image)
        contour = cls.get_barcode_contours(dilated_image)

        if rgb:
            return cls.draw_detected_contour(gray_img,
                                             contour,
                                             contour_color)

        else:
            return cls.draw_detected_contour(img,
                                             contour,
                                             contour_color)

    @staticmethod
    def get_image_gradient(gray_img):
        # compute the Scharr gradient magnitude representation of the images
        gradX = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        return gradient

    @staticmethod
    def get_threshold_image(gradient_image, threshold):
        # blur and threshold the image
        blurred = cv2.blur(gradient_image, (9, 9))
        (_, thresh) = cv2.threshold(blurred, threshold,
                                    255, cv2.THRESH_BINARY)

        return thresh

    @staticmethod
    def kernel_erode_dilate(threshold_image):
        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        return closed

    @staticmethod
    def get_barcode_contours(dilated_image):
        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        (cnts, _) = cv2.findContours(dilated_image.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        return box

    @staticmethod
    def draw_detected_contour(img, contour, color=(0, 255, 0)):
        # draw a bounding box around the detected barcode and display the image
        cv2.drawContours(img, [contour], -1, color, 2)
        return img


if __name__ == "__main__":
    main()
