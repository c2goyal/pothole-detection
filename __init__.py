try:
    import cv2
    print("OpenCV is installed. Version:", cv2._version_)
except ImportError:
    print("OpenCV is not installed.")