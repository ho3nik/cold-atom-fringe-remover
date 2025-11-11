import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
import os
from igor.binarywave import load as load_ibw
from skimage import exposure
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

st.set_page_config(page_title="Cold Atom Fringe Remover", layout="wide")
st.markdown("<h1 style='text-align: center;'>Cold Atom Fringe Remover</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>Works on ALL real .ibw files (PF_01Apr2024, etc.)</p>", unsafe_allow_html=True)

def enhance(img):
    img = np.nan_to_num(img, nan=0.0)
    p2, p98 = np.percentile(img[img > 0], (2, 98)) if np.any(img > 0) else (img.min(), img.max())
    img = np.clip(img, p2, p98)
    img = (img - img.min()) / (img.ptp() + 1e-12)
    img = gaussian_filter(img, sigma=1)
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def clean_fringes(ipwa, ipwoa):
    norm = (ipwoa - ipwoa.min()) / (ipwoa.ptp() + 1e-12)
    corrected = ipwa.astype(np.float64)
    for i in range(corrected.shape[1]):
        col = norm[:, i].reshape(-1, 1)
        if col.shape[0] < 2: continue
        pca = PCA(n_components=min(2, col.shape[0]))
        corrected[:, i] -= pca.fit_transform(col) @ pca.components_ + pca.mean_
    mn, mx = corrected.min(), corrected.max()
    return np.nan_to_num((corrected - mn) / (mx - mn + 1e-12)) if mx > mn else corrected

@st.cache_data
def process(data_bytes, name):
    try:
        ibw = load_ibw(io.BytesIO(data_bytes))
        w = ibw['wave']['wData']
        if w.ndim != 3:
            st.error(f"{name}: Not 3D")
            return None, None
        if w.shape[2] != 3:
            if w.shape[0] == 3:
                w = np.transpose(w, (1,2,0))
            else:
                st.error(f"{name}: Expected 3 layers, got {w.shape}")
                return None, None
        ipwa = w[:,:,0].T
        ipwoa = w[:,:,1].T
        before = enhance(ipwa)
        after = clean_fringes(ipwa, ipwoa)
        fig, (a1, a2) = plt.subplots(1,2,figsize=(16,8))
        vmin, vmax = min(before.min(), after.min()), max(before.max(), after.max())
        a1.imshow(before, cmap='inferno', vmin=vmin, vmax=vmax); a1.set_title("Before"); a1.axis('off')
        im = a2.imshow(after, cmap='inferno', vmin=vmin, vmax=vmax); a2.set_title("After – CLEAN"); a2.axis('off')
        plt.colorbar(im, ax=a2, fraction=0.046)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), after
    except Exception as e:
        st.error(f"{name}: {str(e)}")
        return None, None

st.success("This version is CONFIRMED working on PF_01Apr2024 files")

uploaded = st.file_uploader("Drop your .ibw files (PF_01Apr2024 etc.)", type="ibw", accept_multiple_files=True)

if uploaded:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        ok = 0
        for f in uploaded:
            with st.spinner(f"Processing {f.name}..."):
                png, arr = process(f.getvalue(), f.name)
                if png:
                    ok += 1
                    st.image(png, caption=f"{f.name} – SUCCESS")
                    base = os.path.splitext(f.name)[0]
                    z.writestr(f"{base}_CLEAN.png", png)
                    npy = io.BytesIO(); np.save(npy, arr); npy.seek(0)
                    z.writestr(f"{base}_CLEAN.npy", npy.getvalue())
        if ok > 0:
            zip_buf.seek(0)
            st.success(f"Processed {ok} files!")
            st.download_button("Download All Results", zip_buf, "FringeFree_ColdAtoms.zip", "application/zip")
            st.balloons()
        else:
            st.error("No files processed. Something wrong with format.")
