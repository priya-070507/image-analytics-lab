import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MODEL=None
def _model():
    global MODEL
    if MODEL is None:
        from keras_facenet import FaceNet
        MODEL=FaceNet()
    return MODEL

def _face(image):
    rgb=np.array(image); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); d=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'); fs=d.detectMultiScale(gray,1.05,5,minSize=(40,40))
    if len(fs)==0:return None
    x,y,w,h=max(fs,key=lambda r:int(r[2])*int(r[3])); return cv2.resize(rgb[y:y+h,x:x+w],(160,160))

def _plot(a,b):
    n=min(128,len(a)); fig,ax=plt.subplots(figsize=(9,3)); ax.plot(a[:n],label='Image 1'); ax.plot(b[:n],label='Image 2'); ax.set_title('First 128 FaceNet embedding dimensions'); ax.set_xlabel('Dimension'); ax.set_ylabel('Value'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.canvas.draw(); arr=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(); plt.close(fig); return Image.fromarray(arr)

def run(image1,image2,threshold=.70):
    f1,f2=_face(image1),_face(image2)
    if f1 is None:return {'ok':False,'message':'Unable to detect a face in Image 1.'}
    if f2 is None:return {'ok':False,'message':'Unable to detect a face in Image 2.'}
    model=_model(); a=model.embeddings(np.expand_dims(f1,0))[0]; b=model.embeddings(np.expand_dims(f2,0))[0]; sim=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))); same=sim>=threshold
    return {'ok':True,'face1':Image.fromarray(f1),'face2':Image.fromarray(f2),'plot':_plot(a,b),'similarity':sim,'same':same,'explanation':f'FaceNet maps each face to an embedding vector. Cosine similarity compares the two vectors. With the selected threshold of {threshold:.2f}, this pair is classified as '+('the same person.' if same else 'different people.')}
