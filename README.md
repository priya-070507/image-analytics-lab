# 🔬 Image Analytics Lab — Image & Video Analytics

Interactive Streamlit academic project covering Template Matching, Viola–Jones, DeepFace and FaceNet.

## Run locally

Use Python 3.11:

```bash
py -3.11 -m venv .venv
.venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

Each experiment follows:

**Input → Preprocessing → Algorithm → Intermediate Visualization → Final Result → Explanation**

DeepFace and FaceNet are imported lazily so the application can start without immediately loading their heavy TensorFlow models.

## Streamlit Cloud

Push the folder to GitHub, create a Streamlit Community Cloud app, select the repository/branch and set the main file to `app.py`. Python 3.11 is requested through `runtime.txt`.

Heavy TensorFlow/DeepFace/FaceNet models may require more memory than some free cloud runtimes provide, so local testing is recommended before deployment.
