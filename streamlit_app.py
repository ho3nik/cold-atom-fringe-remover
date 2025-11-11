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
st.markdown("<h1 style='text-align: center;'>🧊 Cold Atom Fringe Remover</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Developed by H. RJ Nikzat</p>", unsafe_allow_html=True)

def enhance(img):
    img = np.nan_to_num(img, nan=0.0)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    img = gaussian_filter(img, sigma=1)
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def clean_fringes(ipwa, ipwoa):
    """
    Removes fringes using PCA. Skips PCA if data is too small.
    """
    norm = (ipwoa - ipwoa.min()) / (np.ptp(ipwoa) + 1e-12)
    corrected = ipwa.astype(float)

    for i in range(ipwa.shape[1]):
        col = norm[:, i].reshape(-1, 1)

        # Only run PCA if there are at least 2 samples
        if col.shape[0] < 2:
            # Not enough data for PCA; skip this column
            continue

        # Determine number of components (can't exceed min(n_samples, n_features))
        n_components = min(2, col.shape[0], col.shape[1])
        if n_components < 1:
            continue

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(col)
        reconstructed = pca.inverse_transform(transformed)
        corrected[:, i] -= reconstructed.flatten()

    return (corrected - corrected.min()) / (np.ptp(corrected) + 1e-12)


def _try_split_2d_into_pair(w):
    h, wcol = w.shape
    if wcol % 2 == 0:
        left = w[:, : wcol // 2]
        right = w[:, wcol // 2 :]
        return left.T, right.T
    if h % 2 == 0:
        top = w[: h // 2, :]
        bottom = w[h // 2 :, :]
        return top.T, bottom.T
    return None

def _extract_wdata(data):
    if isinstance(data, dict):
        wave = data.get('wave') or data.get('w')
        if isinstance(wave, dict) and 'wData' in wave:
            return wave['wData']
        if isinstance(wave, np.ndarray):
            return wave
        if 'wData' in data:
            return data['wData']
    if isinstance(data, np.ndarray):
        return data
    return None

@st.cache_data
def process(bytes_data, name):
    try:
        data = load_ibw(io.BytesIO(bytes_data))
        w = _extract_wdata(data)
        if w is None:
            st.error(f"{name}: Couldn't find numeric wave data in file structure.")
            return None, None

        w = np.asarray(w)
        w = np.squeeze(w)

        if w.ndim == 3:
            if w.shape[2] in (2,3):
                ipwa = w[:, :, 0].T
                ipwoa = w[:, :, 1].T
            elif w.shape[0] in (2,3):
                w = np.transpose(w, (1, 2, 0))
                ipwa = w[:, :, 0].T
                ipwoa = w[:, :, 1].T
            elif w.shape[0] >= 2:
                ipwa = w[0, :, :].T
                ipwoa = w[1, :, :].T
            else:
                st.error(f"{name}: Unexpected 3D shape {w.shape}")
                return None, None
        elif w.ndim == 2:
            maybe = _try_split_2d_into_pair(w)
            if maybe is not None:
                ipwa, ipwoa = maybe
            else:
                h, wc = w.shape
                if wc > h and wc % 2 == 1 and wc - 1 > 0:
                    left = w[:, : (wc - 1) // 2]
                    right = w[:, (wc - 1) // 2 + 1 :]
                    if left.shape == right.shape:
                        ipwa = left.T
                        ipwoa = right.T
                    else:
                        st.error(f"{name}: 2D image but cannot split into a pair (shape {w.shape}).")
                        return None, None
                else:
                    st.error(f"{name}: Not 3D data (found 2D shape {w.shape}).")
                    return None, None
        else:
            st.error(f"{name}: Unsupported data dimensionality: {w.ndim} (shape {w.shape})")
            return None, None

        if ipwa.size == 0 or ipwoa.size == 0:
            st.error(f"{name}: Extracted empty image(s).")
            return None, None
        if ipwa.shape != ipwoa.shape:
            if ipwa.T.shape == ipwoa.shape:
                ipwa = ipwa.T
            else:
                st.error(f"{name}: Paired images have different shapes: {ipwa.shape} vs {ipwoa.shape}")
                return None, None

        before = enhance(ipwa)
        after = clean_fringes(ipwa, ipwoa)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))
        vmin, vmax = min(before.min(), after.min()), max(before.max(), after.max())
        a1.imshow(before, cmap='inferno', vmin=vmin, vmax=vmax)
        a1.set_title("Before")
        a1.axis('off')
        im = a2.imshow(after, cmap='inferno', vmin=vmin, vmax=vmax)
        a2.set_title("After – CLEAN")
        a2.axis('off')
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

uploaded = st.file_uploader("Drop your .ibw files here", type="ibw", accept_multiple_files=True)

if uploaded:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        success = 0
        for f in uploaded:
            with st.spinner(f"Cleaning {f.name}..."):
                png, arr = process(f.getvalue(), f.name)
                if png:
                    success += 1
                    st.image(png, caption=f"✅ {f.name}")
                    base = os.path.splitext(f.name)[0]
                    z.writestr(f"{base}_CLEAN.png", png)
                    npy = io.BytesIO()
                    np.save(npy, arr)
                    npy.seek(0)
                    z.writestr(f"{base}_CLEAN.npy", npy.getvalue())
        if success:
            zip_buf.seek(0)
            st.success(f"Processed {success} files!")
            st.download_button("📦 Download All", zip_buf, "FringeFree_ColdAtoms.zip", "application/zip")
            st.balloons()
        else:
            st.error("No files worked. Contact me.")
