from __future__ import annotations

import numpy as np

# Created by storm at 21.10.24

from flask import Flask, render_template, request, jsonify
from change_testmodel_depart_coefficient import change_depart_coefficient, rerun_new_depart_coefficient
import plotly.graph_objs as go
from convert_grid_to_ts import compute_tau_scale

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    abund = float(data['abund'])
    x_values_lower = data['x_lower']
    y_values_lower = data['y_lower']
    x_values_upper = data['x_upper']
    y_values_upper = data['y_upper']

    save_path = "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/save/"
    dtype = "<f4"

    with open(save_path + "atom_patch_meta.txt", "r") as mesh:
        mesh.readline()
        _, _, _, _, _, dims_atom_levels, _, _ = [i for i in mesh.readline().split()]
        atom_levels = int(dims_atom_levels)

    with open(save_path + "../atmos_mesh.txt", "r") as mesh:
        nx, ny, nz = [int(i) for i in mesh.readline().split()]
        dx, dy, dz = [float(i) for i in mesh.readline().split()]
    zz = np.arange(nz) * dz

    atmo_dimension = nz

    with open(save_path + "atmos_001000.bin", "rb") as atmos_file:
        # 32         1         1       256         7         1
        tau = np.fromfile(atmos_file, dtype=dtype).reshape(atmo_dimension, 7, order="F")[
              :, -1
              ]

        tau = compute_tau_scale(tau, zz * 1e8, axis=-1)
        log_tau = np.log10(tau)
        # print all the data in the file
   # print(log_tau)

    # interpolate 1D as a function of equally spaced x_values, y_values
    new_y_values_lower = np.interp(np.linspace(np.min(x_values_lower), np.max(x_values_lower), atmo_dimension), x_values_lower, y_values_lower)
    new_y_values_upper = np.interp(np.linspace(np.min(x_values_upper), np.max(x_values_upper), atmo_dimension), x_values_upper, y_values_upper)

    change_depart_coefficient(save_path, new_y_values_lower, new_y_values_upper, atmo_dimension, atom_levels)
    xx_lte, yy_lte, xx_nlte, yy_nlte, ltau_lte, cf_lte, ltau_nlte, cf_nlte = rerun_new_depart_coefficient("/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/",
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/test_precomp/",
                                                                    "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/atom.txt",
                                                                    abund,
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/",
                                                                    4553.6, 4554.5)
    fig = create_plot_data(xx_lte, yy_lte, xx_nlte, yy_nlte, "Wavelength [Å]", "Continuum flux")
    fig2 = create_plot_data(ltau_lte, cf_lte, ltau_nlte, cf_nlte, "log τ", "Contribution function")
    return jsonify({"data": fig.to_json(), "data2": fig2.to_json()})
    # Process the data as needed
    #result = your_processing_function(data)
    #return jsonify(result)

def create_plot_data(x_fitted, y_fitted, x_obs, y_obs, xaxis_title, yaxis_title):
    # plot fitted as line
    trace = go.Scatter(x=list(x_fitted), y=list(y_fitted), mode='lines', line=dict(color='red'), name='LTE')
    # plot observed data as a scatter plot
    trace_obs = go.Scatter(x=list(x_obs), y=list(y_obs), mode='lines', marker=dict(color='black'), name='NLTE')
    # xlimit is the range of x values to plot
    xlimit = [min(x_obs), max(x_obs)]
    # find y_fitted that is within xlimit
    y_fitted2 = y_fitted[(x_fitted >= xlimit[0]) & (x_fitted <= xlimit[1])]
    if np.size(y_fitted2) > 0:
        max_y = max(max(y_fitted2) + 0.03, 1.03)
        ylimit = min(y_fitted2) - 0.03, max_y
    #else:
    ylimit = 0, 1.03
    fig = go.Figure(data=[trace_obs, trace], layout_xaxis_range=xlimit, layout_yaxis_range=ylimit)
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
