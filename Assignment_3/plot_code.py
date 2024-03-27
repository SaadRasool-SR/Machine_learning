import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots



def score_plot(data_list, y_data_list, title, x_title, y_title, labels_list):
    fig = go.Figure()

    for i in range(len(data_list)):
        # Add traces
        fig.add_trace(go.Scatter(x=y_data_list[i], y=data_list[i],
                            mode='lines+markers',
                            name=labels_list[i]))

    fig.update_layout(title=title)
    fig.update_xaxes(title = x_title)
    fig.update_yaxes(title = y_title)
    fig.show()

def new_score_plot(x_data_list, y_data_list, title, x_title, y_title, labels_list):
    fig = go.Figure()

    for i in range(len(x_data_list)):
        # Add traces
        fig.add_trace(go.Scatter(x=x_data_list[i], y=y_data_list[i],
                            mode='lines+markers',
                            name=labels_list[i]))

    fig.update_layout(title=title)
    fig.update_xaxes(title = x_title)
    fig.update_yaxes(title = y_title)
    fig.show()

def new_score_plot_2(projections, target):
    fig = px.scatter(
    projections, x=0, y=1,
    color=target, labels={'color': 'target'}
    )
    fig.update_layout(
        title="t-SNE Visualization (Heart Disease Dataset)",
        xaxis_title="First Feature",
        yaxis_title="Second Feature"
    )
    fig.show()

def new_score_plot_3(projections, target):
    fig = px.scatter(
    projections, x=0, y=1,
    color=target, labels={'color': 'target'},
    size_max=10,
    color_discrete_map={0: 'red', 1: 'black'}
    )
    fig.update_layout(
        title="t-SNE Visualization (Breast Cancer Dataset)",
        xaxis_title="First Feature",
        yaxis_title="Second Feature"
    )
    fig.show() 
 


def bar_score_plot(x, y, title, x_title, y_title, y_cum):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create bar plot
    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        marker_color='blue',  # Change color if needed
        name=y_title,
    ))

    fig.add_trace(go.Scatter(x=x,
                            y=y_cum,
                            mode='lines+markers',
                            name='Cumulative % Variation',
                            yaxis='y2'))

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, side = 'left', showgrid=False),
        yaxis2=dict(title='Cumulative % Variation', overlaying='y', side='right', showgrid=False)
    )

    fig.show()

def bar_plot(x,y,title,x_title,y_title):

      # Create bar plot
    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        marker_color='blue',  # Change color if needed
        name=y_title,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, side = 'left')
    )

    fig.show()


def sns_pairplot(data):
    return sns.pairplot(data)


#def DBSCAN