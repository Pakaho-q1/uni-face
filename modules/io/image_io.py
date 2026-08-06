import cv2

class ImageIO:
    @staticmethod
    def read(filepath: str):
        return cv2.imread(filepath)

    @staticmethod
    def write(filepath: str, image):
        cv2.imwrite(filepath, image)
