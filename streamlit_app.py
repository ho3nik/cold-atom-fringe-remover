# streamlit_app.py
import streamlit as st
st.set_page_config(page_title="Cold Atom Fringe Remover", layout="wide")

# Force light mode + big title
st.markdown("""
<style>
    .css-1y0ar0j {background-color: #ffffff !important;}
    .css-1d391kg {padding-top: 1rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("Cold Atom Fringe Remover")
st.markdown("**Upload .ibw files → Get fringe-free atoms in 3 seconds**")
st.markdown("---")

import numpy as np, matplotlib.pyplot as plt, io, zipfile, os
from igor2.binarywave import load as load_ibw
from skimage import exposure
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

def enhance(img):
    img = np.nan_to_num(img)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - img.min())/(img.max()-img.min()+1e-8)
    img = gaussian_filter(img, sigma=1)
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def clean_fringes(ipwa, ipwoa):
    norm = (ipwoa - ipwoa.min())/(ipwoa.max()-ipwoa.min()+1e-8)
    corrected = np.zeros_like(ipwa)
    for i in range(ipwa.shape[1]):
        col = norm[:, i].reshape(-1, 1)
        n = min(2, col.shape[0])
        if n > 0:
            pca = PCA(n_components=n)
            rec = pca.fit_transform(col)
            rec = pca.inverse_transform(rec).flatten()
            corrected[:, i] = ipwa[:, i] - rec
    return (corrected - corrected.min())/(corrected.max()-corrected.min()+1e-8)

@st.cache_data
def process(f_bytes, name):
    data = load_ibw(io.BytesIO(f_bytes))
    w = data['wave']['wData']
    if w.ndim != 3 or w.shape[2] != 3:
        return None, None
    ipwa = w[:,:,0].T
    ipwoa = w[:,:,1].T
    before = enhance(ipwa)
    after = clean_fringes(ipwa, ipwoa)

    fig, (a1, a2) = plt.subplots(1,2,figsize=(15,7))
    vmin, vmax = min(before.min(), after.min()), max(before.max(), after.max())
    a1.imshow(before, cmap='inferno', vmin=vmin, vmax=vmax); a1.axis('off'); a1.set_title("Before")
    im = a2.imshow(after, cmap='inferno', vmin=vmin, vmax=vmax); a2.axis('off'); a2.set_title("After – CLEAN!")
    plt.colorbar(im, ax=a2, fraction=0.046)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue(), io.BytesIO()

uploaded = st.file_uploader("Drop your .ibw files here", type="ibw", accept_multiple_files=True)

if uploaded:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for f in uploaded:
            with st.spinner(f"Cleaning {f.name}..."):
                png, _ = process(f.getvalue(), f.name)
                if png:
                    st.image(png, caption=f"✅ {f.name}")
                    z.writestr(f"{os.path.splitext(f.name)[0]}_CLEAN.png", png)
    zip_buf.seek(0)
    st.download_button("Download ALL Cleaned Images", zip_buf, "FringeFree_ColdAtoms.zip", "application/zip")
    st.balloons()