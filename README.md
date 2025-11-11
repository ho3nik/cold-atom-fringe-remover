# 🧊 Cold Atom Fringe Remover
A web application to enhance and remove interference fringes from cold atom images stored in `.ibw` files. Developed by **H. RJ Nikzat**.

---

## 🌐 Live App

Try the app online: [https://cold-atom-fringe-remover-7eznbcc3gexhav8qcpsfvm.streamlit.app/]

## ⚡ Features

- Automatic enhancement of cold atom images
- Fringe removal using PCA (when data allows)
- Handles common `.ibw` file structures:
  - 3D arrays with atom + reference images
  - 2D paired images (side-by-side or top-bottom)
- Outputs cleaned images (`.png`) and numerical arrays (`.npy`)
- Zip download of processed files

---

## 📝 Supported `.ibw` Files

- **Works best:** Paired images with sufficient rows and columns for PCA processing.
- **May not work:** Single-channel or unusually structured files; PCA requires at least 2 rows per column.

---
