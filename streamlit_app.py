import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
import os
from igor2.binarywave import load as load_ibw
from skimage import exposure
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

st.set_page_config(page_title="Cold Atom Fringe Remover", layout="wide")
st.markdown("<h1 style='text-align: center;'>Cold Atom Fringe Remover</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>Upload .ibw → Get clean atoms instantly</p>", unsafe_allow_html=True)

def enhance(img):
    img = np.nan_to_num(img)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    img = gaussian_filter(img, sigma=1)
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def clean_fringes(ipwa, ipwoa):
    norm = (ipwoa - ipwoa.min()) / (ipwoa.max() - ipwoa.min() + 1e-12)
    corrected = ipwa.astype(np.float64).copy()
    for i in range(ipwa.shape[1]):
        col = norm[:, i].reshape(-1, 1)
        if col.shape[0] < 2:
            continue
        pca = PCA(n_components=min(2, col.shape[0]))
        rec = pca.fit_transform(col)
        rec = pca.inverse_transform(rec).flatten()
        corrected[:, i] -= rec
    mn, mx = corrected.min(), corrected.max()
    return (corrected - mn) / (mx - mn + 1e-12) if mx > mn else corrected

@st.cache_data
def process(bytes_data, filename):
    try:
        data = load_ibw(io.BytesIO(bytes_data))
        w = data['wave']['wData']
        if w.ndim != 3 or w.shape[2] != 3:
            st.error(f"{filename}: Not a valid 3-layer .ibw file")
            return None, None
        
        ipwa  = w[:,:,0].T  # atoms + probe
        ipwoa = w[:,:,1].T  # probe only
        # idark = w[:,:,2].T  # not used yet

        before = enhance(ipwa)
        after = clean_fringes(ipwa, ipwoa)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        vmin = min(before.min(), after.min())
        vmax = max(before.max(), after.max())
        ax1.imshow(before, cmap='inferno', vmin=vmin, vmax=vmax)
        ax1.set_title("Before – With Fringes")
        ax1.axis('off')
        im = ax2.imshow(after, cmap='inferno', vmin=vmin, vmax=vmax)
        ax2.set_title("After – Fringes GONE!")
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), after
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return None, None

uploaded = st.file_uploader("**Drop your .ibw files here**", type="ibw", accept_multiple_files=True)

if uploaded:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for file in uploaded:
            with st.spinner(f"Cleaning {file.name}..."):
                png_bytes, cleaned_array = process(file.getvalue(), file.name)
                if png_bytes:
                    st.image(png_bytes, caption=f"Success: {file.name}")
                    name = os.path.splitext(file.name)[0]
                    z.writestr(f"{name}_CLEAN.png", png_bytes)
                    # Also save as .npy for real analysis
                    np_buf = io.BytesIO()
                    np.save(np_buf, cleaned_array)
                    np_buf.seek(0)
                    z.writestr(f"{name}_CLEAN.npy", np_buf.getvalue())
                else:
                    st.warning(f"Failed: {file.name}")

    zip_buf.seek(0)
    if zip_buf.getbuffer().nbytes > 1000:  # not empty
        st.success("ALL DONE! Download your clean images + data")
        st.download_button(
            "Download FULL Results (PNG + NumPy .npy)",
            zip_buf,
            "FringeFree_ColdAtoms.zip",
            "application/zip"
        )
        st.balloons()
    else:
        st.error("No files were processed successfully. Check your .ibw files have 3 layers.")
