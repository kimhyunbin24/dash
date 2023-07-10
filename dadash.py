import streamlit as st
import pandas as pd
import nibabel as nib
import time
from time import time
import numpy as np
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage
import plotly.graph_objects as go
import plotly.express as px
import urllib.parse
from scipy.ndimage import median_filter

# Load CT meta data
data_raw = pd.read_csv("./research.csv")
data = data_raw

nii_gz_path = "./image/3.MRIT1.1855023.202203220024_T1.nii.gz"
nii_gz = nib.load(nii_gz_path)
num_slices = nii_gz.shape[2]

##########################################################################
# 3d mesh 전처리 부분
img = image.load_img("./image/CT.00282188.20210623.nii.gz")
mat = img.affine
img = img.get_fdata()
img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]
spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])

# med_img = filters.median(img, selem=np.ones((1, 3, 3), dtype=bool))
med_img = median_filter(img, footprint=np.ones((1, 3, 3), dtype=bool))

verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=5)
x, y, z = verts.T
i, j, k = faces.T
fig_mesh = go.Figure()
fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

# 슬라이서
slicer1 = VolumeSlicer(img, axis=0, spacing=spacing, thumbnail=False)

#############################################################################

num_ticks = 10
step_size = max(1, num_slices // num_ticks)
marks = {i: str(i) for i in range(0, num_slices, step_size)}

# Define modal component
def display_patient_info(active_cell, data):
    if active_cell:
        row_index = active_cell["row"]
        column_id = active_cell["column_id"]
        if column_id == "PatientID":
            patient_id = data[row_index][column_id]
            patient_info = data_raw[data_raw["PatientID"] == patient_id][["PatientSex", "PatientAge"]]
            if not patient_info.empty:
                sex = patient_info.iloc[0]["PatientSex"]
                age = patient_info.iloc[0]["PatientAge"]
                return [
                    st.markdown(f"환자 ID: {patient_id}"),
                    st.markdown(f"성별: {sex}"),
                    st.markdown(f"연령: {age}"),
                ]
    return None

@st.cache
def load_nifti_data():
    nii_gz_path = "./image/CT.00282188.20210623.nii.gz"
    nii_gz = nib.load(nii_gz_path)
    data = nii_gz.get_fdata()
    return data

@st.cache
def preprocess_image_data():
    img = image.load_img("./image/CT.00282188.20210623.nii.gz")
    mat = img.affine
    img = img.get_fdata()
    img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]
    spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])

    # med_img = filters.median(img, selem=np.ones((1, 3, 3), dtype=bool))
    med_img = median_filter(img, footprint=np.ones((1, 3, 3), dtype=bool))
    
    return med_img, spacing

def update_segmentation_slices(annotations):
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        mask = np.zeros_like(med_img)
        overlay1 = slicer1.create_overlay_data(mask)
        return go.Mesh3d(), overlay1
    else:
        path = path_to_coords(annotations["z"]["path"])
        rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
        mask = np.zeros(img.shape[1:])
        mask[rr, cc] = 1
        mask = ndimage.binary_fill_holes(mask)
        top, bottom = sorted(
            [int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]]
        )
        img_mask = np.logical_and(med_img > v_min, med_img <= v_max)
        img_mask[:top] = False
        img_mask[bottom:] = False
        img_mask[top:bottom, np.logical_not(mask)] = False
        img_mask = largest_connected_component(img_mask)
        verts, faces, _, _ = measure.marching_cubes(
            filters.median(img_mask, selem=np.ones((1, 7, 7))), 0.5, step_size=3
        )
        x, y, z = verts.T
        i, j, k = faces.T
        trace = go.Mesh3d(x=z, y=y, z=x, color="red", opacity=0.8, i=k, j=j, k=i)
        overlay1 = slicer1.create_overlay_data(img_mask)
        return trace, overlay1


def main():
    st.title("Analysis Viewer")
    
    # Display patient check table
    st.dataframe(data[["PatientID", "PatientSex", "PatientAge"]])
    
    # Display patient modal
    active_cell = st.table(data[["PatientID", "PatientSex", "PatientAge"]]).where(st.button("View Details"))
    patient_info = display
