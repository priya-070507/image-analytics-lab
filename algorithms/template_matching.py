import cv2
import numpy as np
from PIL import Image

def run(source, template):
    s=cv2.cvtColor(np.array(source),cv2.COLOR_RGB2BGR); t=cv2.cvtColor(np.array(template),cv2.COLOR_RGB2BGR)
    gs=cv2.cvtColor(s,cv2.COLOR_BGR2GRAY); gt=cv2.cvtColor(t,cv2.COLOR_BGR2GRAY); h,w=gt.shape
    if h>gs.shape[0] or w>gs.shape[1]: raise ValueError('Template must be smaller than the source image.')
    m=cv2.matchTemplate(gs,gt,cv2.TM_CCOEFF_NORMED); _,score,_,loc=cv2.minMaxLoc(m); x,y=loc
    out=s.copy(); cv2.rectangle(out,(x,y),(x+w,y+h),(122,72,210),3); cv2.putText(out,f'Match: {score:.3f}',(x,max(25,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.8,(122,72,210),2)
    heat=cv2.applyColorMap(cv2.normalize(m,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
    return {'score':float(score),'box':(x,y,w,h),'gray_source':Image.fromarray(gs),'gray_template':Image.fromarray(gt),'heatmap':Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)),'result':Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)),'explanation':'The template is slid across the source image. TM_CCOEFF_NORMED calculates normalized similarity at each location. The highest score is selected as the best match.'}
