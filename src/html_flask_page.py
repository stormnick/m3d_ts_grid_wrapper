from __future__ import annotations

import numpy as np

# Created by storm at 21.10.24

from flask import Flask, render_template, request, jsonify
from change_testmodel_depart_coefficient import change_depart_coefficient, rerun_new_depart_coefficient
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    abund = float(data['abund'])
    x_values = data['x']
    y_values = data['y']

    atmo_dimension = 256
    atom_levels = 6
    save_path = "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/save/"

    # interpolate 1D as a function of equally spaced x_values, y_values
    new_y_values = np.interp(np.linspace(np.min(x_values), np.max(x_values), atmo_dimension), x_values, y_values)

    print(new_y_values)
    change_depart_coefficient(save_path, new_y_values, np.ones_like(atmo_dimension), atmo_dimension, atom_levels)
    xx_lte, yy_lte, xx_nlte, yy_nlte = rerun_new_depart_coefficient("/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/",
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/test_precomp/",
                                                                    "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/atom.txt",
                                                                    abund,
                                                                    "/Users/storm/PycharmProjects/m3d_ts_grid_wrapper/src/input_ba_test_ba6/",
                                                                    4553.6, 4554.5)
    fig = create_plot_data(xx_lte, yy_lte, xx_nlte, yy_nlte)
    return jsonify({"data": fig.to_json()})
    # Process the data as needed
    #result = your_processing_function(data)
    #return jsonify(result)

def create_plot_data(x_fitted, y_fitted, x_obs, y_obs):
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
        xaxis_title="Wavelength",
        yaxis_title="Normalised Flux"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
