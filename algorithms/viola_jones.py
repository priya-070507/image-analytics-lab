import cv2
import numpy as np
from PIL import Image

def run(image,scale_factor=1.05,min_neighbors=6):
    bgr=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY); path=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'; detector=cv2.CascadeClassifier(path)
    faces=detector.detectMultiScale(gray,scaleFactor=float(scale_factor),minNeighbors=int(min_neighbors),minSize=(35,35)); out=bgr.copy()
    for x,y,w,h in faces: cv2.rectangle(out,(x,y),(x+w,y+h),(122,72,210),3)
    integral=cv2.normalize(cv2.integral(gray)[1:,1:],None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    sob=cv2.Sobel(cv2.GaussianBlur(gray,(3,3),0),cv2.CV_32F,1,0,ksize=3); haar=cv2.normalize(abs(sob),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    return {'faces':faces.tolist(),'gray':Image.fromarray(gray),'haar':Image.fromarray(haar),'integral':Image.fromarray(integral),'result':Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)),'explanation':'Viola–Jones uses Haar-like rectangular features. Integral images make feature evaluation efficient, while the cascade rejects non-face regions in stages.'}
