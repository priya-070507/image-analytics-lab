import cv2
import numpy as np
from PIL import Image

def _scores(x):
    return {str(k):round(float(v),3) for k,v in (x or {}).items() if isinstance(v,(int,float,np.integer,np.floating))}

def run(image):
    from deepface import DeepFace
    rgb=np.array(image); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    detected=DeepFace.extract_faces(img_path=bgr,detector_backend='opencv',enforce_detection=False,align=True)
    valid=[x for x in detected if x.get('face') is not None and float(x.get('confidence',1))>0]
    view=bgr.copy(); results=[]
    for item in valid:
        reg=item.get('facial_area',{}) or {}; x,y,w,h=[int(reg.get(k,0)) for k in ('x','y','w','h')]
        if w and h: cv2.rectangle(view,(x,y),(x+w,y+h),(122,72,210),3)
        face=np.asarray(item['face']); face=np.clip(face*255 if face.dtype!=np.uint8 else face,0,255).astype(np.uint8)
        try:
            a=DeepFace.analyze(img_path=face,actions=['age','gender','emotion','race'],detector_backend='skip',enforce_detection=False,silent=True)
            a=a[0] if isinstance(a,list) else a
            results.append({'age':int(round(float(a.get('age',0)))),'gender':str(a.get('dominant_gender','Unknown')),'emotion':str(a.get('dominant_emotion','Unknown')),'race':str(a.get('dominant_race','Unknown')),'gender_scores':_scores(a.get('gender',{})),'emotion_scores':_scores(a.get('emotion',{})),'race_scores':_scores(a.get('race',{}))})
        except Exception:
            results.append({'age':0,'gender':'Unavailable','emotion':'Unavailable','race':'Unavailable','gender_scores':{},'emotion_scores':{},'race_scores':{}})
    crop=valid[0]['face'] if valid else rgb; crop=np.asarray(crop); crop=np.clip(crop*255 if crop.dtype!=np.uint8 else crop,0,255).astype(np.uint8)
    return {'faces':results,'detection':Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)),'crop':Image.fromarray(crop),'explanation':'DeepFace detects facial regions and applies deep-learning models for facial attribute estimation. These outputs are model predictions and may be incorrect.'}
