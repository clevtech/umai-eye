import requests
import cv2
url = 'http://34.125.120.156/object-to-img'
files = {'media': open('1234.jpg', 'rb')}
r=cv2.imread(requests.get(url, files=files))
cv2.imshow("test",r)