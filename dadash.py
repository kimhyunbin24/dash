from dash import dash_table
from dash import html
from dash import dcc
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
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash_slicer import VolumeSlicer
import urllib.parse

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash_slicer import VolumeSlicer
from scipy.ndimage import median_filter
import numpy as np

app = Dash(__name__)


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server

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
slicer1 = VolumeSlicer(app, img, axis=0, spacing=spacing, thumbnail=False)

#############################################################################

num_ticks = 10
step_size = max(1, num_slices // num_ticks)
marks = {i: str(i) for i in range(0, num_slices, step_size)}

# Define modal component
modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("환자 정보"),
                dbc.ModalBody(
                    [
                        html.Div(id="patient-modal-body"),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="image-graph-2d"), width=6),
                                dbc.Col(
                                    [
                                        dcc.Graph(id="image-graph-3d", figure=fig_mesh),
                                        html.P("3D mesh representation of the image data and annotation")
                                    ],
                                    width=6,
                                ),
                            ],
                            justify="center",
                            align="center",
                            style={"margin": "10px"},
                        ),
                        dcc.Slider(
                            id="slice-slider",
                            min=0,
                            max=num_slices - 1,
                            step=1,
                            value=0,
                            marks=marks,
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button("닫기", id="close-modal", className="ml-auto", n_clicks=0)
                ),
            ],
            id="patient-modal",
            size="xl",
            centered=True,
        ),
    ]
)

mesh_card = dbc.Card(
    [
        dbc.CardHeader("3D CT image"),
        dbc.CardBody([dcc.Graph(id="graph-helper", figure=fig_mesh)]),
    ]
)

# Define app layout
app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dash_table.DataTable(
                                id="patient_check",
                                columns=[{"name": i, "id": i, "selectable": True} for i in data[["PatientID", "PatientSex", "PatientAge"]].columns],
                                data=data[["PatientID", "PatientSex", "PatientAge"]].to_dict("records"),
                                style_cell={
                                    "fontFamily": "Arial, sans-serif",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "textAlign": "center",
                                    "fontSize": "14px",
                                    "padding": "0px",
                                },
                                style_table={
                                    "fontFamily": "Arial, sans-serif",
                                    "width": "300px",
                                    "minWidth": "50%",
                                    "maxWidth": "300px",
                                    "overflowX": "scroll",
                                    "overflowY": "scroll",
                                    "maxHeight": "500px",
                                    "margin": "0px",
                                },
                                editable=True,
                                filter_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                row_selectable="multi",
                                row_deletable=False,
                                selected_rows=[],
                                page_action="none",
                                page_current=0,
                                page_size=10,
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            html.Div(
                                id="analysis-container",
                                style={"margin": "0px"},
                                children=[
                                    html.H3("Analysis Viewer"),
                                    dash_table.DataTable(
                                        id="analysis_viewer",
                                        columns=[
                                            {"name": i, "id": i, "selectable": False, "deletable": True}
                                            for i in data_raw.columns
                                        ],
                                        style_cell={
                                            "fontFamily": "Arial, sans-serif",
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                            "textAlign": "center",
                                            "fontSize": "14px",
                                            "padding": "0px",
                                        },
                                        style_table={
                                            "fontFamily": "Arial, sans-serif",
                                            "width": "700px",
                                            "minWidth": "50%",
                                            "maxWidth": "700px",
                                            "overflowX": "scroll",
                                            "overflowY": "scroll",
                                            "maxHeight": "500px",
                                        },
                                        data=[],
                                        editable=True,
                                        filter_action="native",
                                        sort_action="native",
                                        sort_mode="multi",
                                        row_selectable=False,
                                        row_deletable=False,
                                        selected_rows=[],
                                        page_action="none",
                                        page_current=0,
                                        page_size=10,
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
                                        ],
                                    ),
                                ],
                            ),
                            width=8,
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Button(
                                    "CSV로 내보내기",
                                    id="export-button",
                                    n_clicks=0,
                                    className="btn btn-primary",
                                    style={"marginTop": "10px", "marginBottom": "10px", "float": "right"},
                                ),
                                width={"size": 8, "offset": 4},
                                align="end",
                            )
                        ),

                    ]

                ),
            ],
            fluid=True,
        ),
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
        modal,

    ]
)

@app.callback(
    Output("patient-modal", "is_open"),
    Output("close-modal", "n_clicks"),
    Input("patient_check", "active_cell"),
    Input("close-modal", "n_clicks"),
    State("patient-modal", "is_open"),
)
def toggle_modal(active_cell, close_modal_n_clicks, is_open):
    if active_cell:
        if active_cell["column_id"] == "PatientID":
            return not is_open, 0
    return False, close_modal_n_clicks


@app.callback(
    Output("patient-modal-body", "children"),
    Input("patient_check", "active_cell"),
    State("patient_check", "data"),
)
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
                    html.P(f"환자 ID: {patient_id}"),
                    html.P(f"성별: {sex}"),
                    html.P(f"연령: {age}"),
                ]
    return None


@app.callback(
    Output("image-graph-2d", "figure"),
    Input("slice-slider", "value")
)
def update_image_graph(slice_idx):
    nii_gz_path = "./image/CT.00282188.20210623.nii.gz"
    nii_gz = nib.load(nii_gz_path)
    data = nii_gz.get_fdata()

    slice_data = data[:, :, slice_idx]

    fig_2d = px.imshow(slice_data, binary_string=True)
    fig_2d.update_layout(
        title="CT Image",
    )

    return fig_2d


@app.callback(
    [
        Output("occlusion-surface", "data"),
        Output(slicer1.overlay_data.id, "data"),
    ],
    [Input("annotations", "data")],
)
def update_segmentation_slices(annotations):
    ctx = dash.callback_context
    if (
        ctx.triggered[0]["prop_id"] == "annotations.data"
        or annotations is None
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


################### patient check에서 오른쪽으로 보내기
@app.callback(
    Output("analysis_viewer", "data"),
    Output("analysis_viewer", "selected_rows"),
    Input("patient_check", "selected_rows"),
)
def update_analysis_viewer(selected_rows):
    if selected_rows is not None:
        selected_data = [data.iloc[idx] for idx in selected_rows]
        selected_data = pd.DataFrame(selected_data, columns=data.columns)
        return selected_data.to_dict('records'), []
    return [], []

##############  csv
@app.callback(
    Output("export_csv", "data"),
    Input("export-button", "n_clicks"),
    State("analysis_viewer", "data"),
)
def export_csv(n_clicks, analysis_viewer_data):
    if n_clicks:
        df = pd.DataFrame(analysis_viewer_data)
        csv_string = df.to_csv(index=False, encoding="utf-8-sig")
        csv_data = "data:text/csv;charset=utf-8-sig," + urllib.parse.quote(csv_string)
        return csv_data
    else:
        return None


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050)

