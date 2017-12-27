import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot

# only needed for saving plot to disk
#import plotly.plotly as py
#py.sign_in('hhhhha', 'm84AvATsSmzvI7XG4EoI')

colorscale = [
    [0, 'white'],
    [0.1, 'red'],
    [1, 'green']
]


def plot_piechart(labels, values, colors=None, textinfo='percentage'):
    """
    Plots a pie-chart with given parameters.
    All parameters must be the same length and order, except 'textinfo'.

    :param labels:      labels of values
    :param values:      values to plot
    :param colors:      colors of each value
    :param textinfo:    'percentage' (default) or 'value';
                        defines weather absolut value or % is shown on chart
    :return:            None if error occurs, else nothing
    """
    if len(labels) == 0 or len(values) == 0 or len(labels) != len(values):
        print('ERROR in Plotter.plot_piechart: Invalid input!')
        return
    if colors:
        if len(colors) != len(labels):
            print('ERROR in Plotter.plot_piechart: Invalid color input!')
            return
        else:
            plot([go.Pie(labels=labels, values=values, textinfo=textinfo, marker=dict(colors=colors))])
    else:
        plot([go.Pie(labels=labels, values=values, textinfo=textinfo)])


def plot_heatmap(x, y, z, colorscale=colorscale, xgap=None, ygap=None):
    if len(x) == 0 or len(y) == 0 or len(z) == 0 or len(x) != len(y) or len(y) != len(z):
        print('ERROR in Plotter.plot_heatmap: Invalid input!')
        print('x:', len(x))
        print('y:', len(y))
        print('z:', len(z))
        return
    fig = None
    if colorscale:
        if xgap and ygap:
            fig = ff.create_annotated_heatmap(z, x, y, colorscale=colorscale, xgap=xgap, ygap=ygap)
        elif xgap:
            fig = ff.create_annotated_heatmap(z, x, y, colorscale=colorscale, xgap=xgap)
        elif ygap:
            fig = ff.create_annotated_heatmap(z, x, y, colorscale=colorscale, ygap=ygap)
        else:
            fig = ff.create_annotated_heatmap(z, x, y, colorscale=colorscale)
    else:
        fig = ff.create_annotated_heatmap(z, x, y)

    # saves figure with low quality and bad formatting
    # py.image.save_as(fig, filename='test.png')
    plot(fig)
