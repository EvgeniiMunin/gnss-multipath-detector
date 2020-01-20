import numpy as np
import matplotlib.pyplot as plt
#import plotly as py
#import plotly.graph_objs as go
#from plotly import plotly as ply

def visualize_plt(data_sample):                                       
    size0, size1 = data_sample.shape[0], data_sample.shape[1] 
    
    # Plot 2D
    plt.imshow(data_sample)
    plt.show()

    # Plot 3D
    fig = plt.figure()
    x = np.linspace(0, size0-1, size0)
    y = np.linspace(0, size1-1, size1)
    print(len(x), len(y))
    print(data_sample.shape)
    #ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    cset = ax.contour3D(x, y, data_sample, 800)
    ax.clabel(cset, fontsize=9, inline=1)
    plt.show()


def visualize_3d_discr(func, 
                       discr_size_fd,
                       scale_code,
                       tau_interv, 
                       dopp_interv, 
                       Tint, 
                       delta_dopp = 0, delta_tau = 0, alpha_att = 1, delta_phase = 0, 
                       filename='3d_surface_check_discr.html'):

    y = np.linspace(tau_interv[0], tau_interv[1], discr_size_fd)
    x = np.linspace(dopp_interv[0], dopp_interv[1], scale_code)
    
    data = [
        go.Surface(
            x = x,
            y = y,
            z = func
        )
    ]

    layout = go.Layout(
        title='3d surface check_discr',
        autosize=True,
        xaxis=go.layout.XAxis(range=[-1000,1000]),
        yaxis=go.layout.YAxis(range=[-1000,1000]),
        scene=dict(
                yaxis=dict(nticks=10,
                           range=[y.min(), y.max()],
                           title='Pixels_X (Code Delay [s])'),
                xaxis=dict(nticks=10,
                           range=[x.min(), x.max()],
                           title='Pixels_Y (Doppler [Hz])'),
                zaxis=dict(nticks=10, range=[func.min(), func.max()]),
                annotations = [dict(
                                    showarrow = False,
                                    x = x.max(), y = y.max(), z = func.max(),
                                    text = 'Tint = {}s'.format(Tint),
                                    xanchor = 'left',
                                    xshift = 10
                ),
                            dict(
                                    showarrow = False,
                                    x = x.max(), y = y.max(), z = func.max()*0.95,
                                    text = 'delta_dopp = {} Hz'.format(delta_dopp),
                                    xanchor = 'left',
                                    xshift = 10
                ),
                            dict(
                                    showarrow = False,
                                    x = x.max(), y = y.max(), z = func.max()*0.9,
                                    text = 'delta_tau = {} s'.format(delta_tau),
                                    xanchor = 'left',
                                    xshift = 10
                ),
                            dict(
                                    showarrow = False,
                                    x = x.max(), y = y.max(), z = func.max()*0.85,
                                    text = 'alpha_att = {}'.format(alpha_att),
                                    xanchor = 'left',
                                    xshift = 10
                ),
                            dict(
                                    showarrow = False,
                                    x = x.max(), y = y.max(), z = func.max()*0.8,
                                    text = 'delta_phase = {} deg'.format(delta_phase * 180 / np.pi),
                                    xanchor = 'left',
                                    xshift = 10
                )
                ]
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename=filename)