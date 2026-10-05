import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title='Image Analytics Lab', page_icon='🔬', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:linear-gradient(135deg,#fbf9ff,#f5f0ff 45%,#fff)}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid #ece7f7}.hero{padding:32px 36px;border-radius:26px;background:linear-gradient(135deg,#eee7ff,#fff);border:1px solid #e5dcfa;margin-bottom:22px}.hero h1{color:#35235f;font-size:42px;font-weight:800;margin:5px 0}.hero p{color:#665a7e;font-size:16px;line-height:1.7}.card{background:#fff;border:1px solid #ebe5f7;border-radius:18px;padding:20px;margin:8px 0;box-shadow:0 8px 25px rgba(71,45,120,.05)}.step{background:#faf8ff;border:1px solid #e9e1f8;border-radius:15px;padding:15px;height:100%}.num{background:#7652c7;color:#fff;border-radius:50%;display:inline-flex;width:29px;height:29px;align-items:center;justify-content:center;font-weight:700}.muted{color:#756b88;font-size:14px;line-height:1.6}.badge{background:#eee7ff;color:#58379a;padding:6px 11px;border-radius:99px;font-size:11px;font-weight:700}.stButton>button{border-radius:11px;background:#7652c7;color:#fff;font-weight:700}.stButton>button:hover{background:#5e3fa6;color:#fff}.stMetric{background:#fff;border:1px solid #e8e0f7;border-radius:15px;padding:8px}
</style>''', unsafe_allow_html=True)

def header(title, subtitle, badge):
    st.markdown(f'<div class="hero"><span class="badge">{badge}</span><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def pipeline(items):
    cols=st.columns(len(items))
    for i,(title,desc) in enumerate(items):
        with cols[i]: st.markdown(f'<div class="step"><span class="num">{i+1}</span><h4>{title}</h4><div class="muted">{desc}</div></div>', unsafe_allow_html=True)

def img(upload): return Image.open(upload).convert('RGB')

def show(title, im): st.markdown(f'**{title}**'); st.image(im,use_container_width=True)

with st.sidebar:
    st.markdown('## 🔬 Image Analytics Lab')
    st.caption('Image & Video Analytics Laboratory')
    page=st.radio('Navigation',['🏠 Dashboard','🎯 Template Matching','👤 Viola–Jones','🧠 DeepFace','🔗 FaceNet','📊 Comparison'])
    st.divider(); st.caption('Input → Preprocessing → Algorithm → Visualization → Result → Explanation')

if page=='🏠 Dashboard':
    header('Image Analytics Lab','An interactive academic laboratory for classical and deep-learning image analytics.','IMAGE & VIDEO ANALYTICS')
    st.markdown('<div class="card"><h3>What is Image Analytics Lab?</h3><div class="muted">Four computer-vision experiments are presented as transparent processing pipelines rather than black boxes. Each page exposes intermediate visualizations and explains the final result.</div></div>',unsafe_allow_html=True)
    st.subheader('Explore the algorithms')
    cols=st.columns(4)
    cards=[('🎯','Template Matching','Locate a template inside a larger image.'),('👤','Viola–Jones','Classical Haar-cascade face detection.'),('🧠','DeepFace','Age, gender, emotion and race analysis.'),('🔗','FaceNet','Face embeddings and verification.')]
    for c,(icon,title,desc) in zip(cols,cards):
        with c: st.markdown(f'<div class="card" style="min-height:180px"><div style="font-size:32px">{icon}</div><h3>{title}</h3><div class="muted">{desc}</div></div>',unsafe_allow_html=True)
    st.subheader('Common architecture'); pipeline([('Input','Upload image(s)'),('Preprocess','Prepare data'),('Algorithm','Run method'),('Visualize','Inspect stages'),('Result','Read output'),('Explain','Understand why')])

elif page=='🎯 Template Matching':
    header('Template Matching','Locate a known template inside a source image using OpenCV correlation.','CLASSICAL COMPUTER VISION')
    pipeline([('Input','Source + template'),('Preprocess','Grayscale'),('Algorithm','TM_CCOEFF_NORMED'),('Intermediate','Similarity map'),('Result','Best match'),('Explain','Interpret score')])
    a,b=st.columns(2); sf=a.file_uploader('Source image',type=['jpg','jpeg','png'],key='tm1'); tf=b.file_uploader('Template image',type=['jpg','jpeg','png'],key='tm2')
    if sf and tf:
        s,t=img(sf),img(tf); a,b=st.columns(2); a.image(s,caption='Source',use_container_width=True); b.image(t,caption='Template',use_container_width=True)
        if st.button('Run Template Matching',type='primary'):
            from algorithms.template_matching import run
            with st.spinner('Running...'): r=run(s,t)
            st.success('Completed'); a,b=st.columns(2); show('Grayscale source',r['gray_source']); show('Grayscale template',r['gray_template']); show('Similarity heatmap',r['heatmap']); show('Final result',r['result'])
            x,y,w,h=r['box']; c1,c2,c3=st.columns(3); c1.metric('Similarity',f"{r['score']:.4f}"); c2.metric('X / Y',f'{x} / {y}'); c3.metric('Size',f'{w} × {h}'); st.info(r['explanation'])
    else: st.info('Upload both images to begin.')

elif page=='👤 Viola–Jones':
    header('Viola–Jones Face Detection','Detect faces using Haar-like features, integral images and a cascade classifier.','CLASSICAL FACE DETECTION')
    pipeline([('Input','Face image'),('Preprocess','Grayscale'),('Features','Haar-like features'),('Integral','Integral image'),('Cascade','Classifier'),('Result','Face boxes')])
    f=st.file_uploader('Upload image',type=['jpg','jpeg','png'],key='vj'); a,b=st.columns(2); scale=a.slider('Scale factor',1.01,1.30,1.05,.01); neigh=b.slider('Minimum neighbors',1,12,6)
    if f:
        im=img(f); show('Input',im)
        if st.button('Run Viola–Jones',type='primary'):
            from algorithms.viola_jones import run
            with st.spinner('Detecting...'): r=run(im,scale,neigh)
            a,b=st.columns(2); show('Grayscale',r['gray']); show('Haar feature visualization',r['haar']); show('Integral image',r['integral']); show('Final detections',r['result']); st.metric('Faces detected',len(r['faces'])); st.info(r['explanation'])
    else: st.info('Upload an image to begin.')

elif page=='🧠 DeepFace':
    header('DeepFace Analysis','Detect faces and estimate age, gender, emotion and race using DeepFace models.','DEEP FACE ANALYSIS')
    pipeline([('Input','Face image'),('Detect','Locate faces'),('Crop','Extract face'),('Analyse','DeepFace models'),('Result','Predictions'),('Explain','Interpret output')])
    st.warning('The first run can be slow because deep-learning models may load or download weights.')
    f=st.file_uploader('Upload face image',type=['jpg','jpeg','png'],key='df')
    if f:
        im=img(f); show('Input',im)
        if st.button('Run DeepFace',type='primary'):
            from algorithms.deepface_analysis import run
            with st.spinner('Loading models and analysing...'): r=run(im)
            if not r['faces']: st.warning('No face detected.')
            else:
                show('Face detection',r['detection']); show('Face crop',r['crop'])
                for i,face in enumerate(r['faces'],1):
                    st.markdown(f'### Face {i}'); a,b,c,d=st.columns(4); a.metric('Age',face['age']); b.metric('Gender',face['gender']); c.metric('Emotion',face['emotion']); d.metric('Race',face['race'])
                    with st.expander('Confidence scores'): st.json({'gender':face['gender_scores'],'emotion':face['emotion_scores'],'race':face['race_scores']})
                st.info(r['explanation'])
    else: st.info('Upload an image to begin.')

elif page=='🔗 FaceNet':
    header('FaceNet Verification','Generate face embeddings and compare two faces using cosine similarity.','FACE EMBEDDINGS')
    pipeline([('Input','Two images'),('Detect','Largest face'),('Preprocess','160 × 160'),('Embed','FaceNet vector'),('Compare','Cosine similarity'),('Result','Decision')])
    st.warning('The first run can be slow because the FaceNet model is loaded on demand.')
    a,b=st.columns(2); f1=a.file_uploader('Image 1',type=['jpg','jpeg','png'],key='f1'); f2=b.file_uploader('Image 2',type=['jpg','jpeg','png'],key='f2'); threshold=st.slider('Similarity threshold',.40,.95,.70,.01)
    if f1 and f2:
        i1,i2=img(f1),img(f2); a,b=st.columns(2); a.image(i1,caption='Image 1',use_container_width=True); b.image(i2,caption='Image 2',use_container_width=True)
        if st.button('Run FaceNet',type='primary'):
            from algorithms.facenet import run
            with st.spinner('Generating embeddings...'): r=run(i1,i2,threshold)
            if not r['ok']: st.error(r['message'])
            else:
                a,b=st.columns(2); show('Face 1',r['face1']); show('Face 2',r['face2']); show('Embedding dimensions',r['plot']); c1,c2,c3=st.columns(3); c1.metric('Cosine similarity',f"{r['similarity']:.4f}"); c2.metric('Percentage',f"{r['similarity']*100:.2f}%"); c3.metric('Threshold',f'{threshold:.2f}')
                (st.success if r['same'] else st.error)(('✅ Same person' if r['same'] else '❌ Different people')); st.info(r['explanation'])
    else: st.info('Upload both images to begin.')

elif page=='📊 Comparison':
    header('Algorithm Comparison','Compare the four techniques from an academic perspective.','COMPARATIVE STUDY')
    st.dataframe(pd.DataFrame({'Algorithm':['Template Matching','Viola–Jones','DeepFace','FaceNet'],'Main task':['Pattern localization','Face detection','Face attribute analysis','Face verification'],'Core idea':['Correlation matching','Haar features + cascade','Deep facial analysis','Face embeddings'],'Output':['Location + score','Bounding boxes','Age / gender / emotion / race','Similarity + decision'],'Type':['Classical CV','Classical CV/ML','Deep learning','Deep learning']}),use_container_width=True,hide_index=True)
    st.markdown('<div class="card"><h3>Key takeaway</h3><div class="muted">Template Matching and Viola–Jones demonstrate classical computer vision. DeepFace and FaceNet demonstrate modern deep-learning based facial analysis and representation.</div></div>',unsafe_allow_html=True)

