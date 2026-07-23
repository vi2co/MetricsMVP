import cv2
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier()

image = cv2.imread("car.jpg")

result = classifier.classify(image)

print(result)