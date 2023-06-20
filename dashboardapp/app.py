# -*- coding: utf-8 -*-
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash.dash_table.Format import Format
from plotly.graph_objs import Layout
from plotly.subplots import make_subplots
from scipy.stats import norm, spearmanr

import about

"""
==========================================================================
App initialization / general info
"""

app_description = """
Using Multiverse Analysis for Estimating Response Models Thesis Dashboard"""
app_title = "Multiverse Analysis for Estimating Response Models Visualizer"
app_image = "https://www.multiverseanalysis.app/assets/app.png" #update this to own image

metas = [ # this should all be updated!!!
    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    {"property": "twitter:card", "content": "summary_large_image"},
    {"property": "twitter:url", "content": "https://www.multiverseanalysis.app/"},
    {"property": "twitter:title", "content": app_title},
    {"property": "twitter:description", "content": app_description},
    {"property": "twitter:image", "content": app_image},
    {"property": "og:title", "content": app_title},
    {"property": "og:type", "content": "website"},
    {"property": "og:description", "content": app_description},
    {"property": "og:image", "content": app_image},
]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME],
    meta_tags=metas,
    title=app_title,
)

server = app.server

"""
==========================================================================
Data sets used in the dashboard
"""


### ----- test data ----- ###

# df_imp_m1 = pd.read_csv("testdata/importance_model1.csv")
# df_imp_m2 = pd.read_csv("testdata/importance_model2.csv")
# df_imp_m3 = pd.read_csv("testdata/importance_model3.csv")
#
# df_perf_m1 = pd.read_csv("testdata/performance_model1.csv")
# df_perf_m2 = pd.read_csv("testdata/performance_model2.csv")
# df_perf_m3 = pd.read_csv("testdata/performance_model3.csv")

### ----- end test data ----- ###

### ----- final data ----- ###

# original (cleaned) data
df_original = pd.read_csv("~/PycharmProjects/ADSThesis/ESS8 data/ESS8_subset_cleaned_timeadj_wmissingvals.csv")

# feature importance RF, SVM and MLP
df_imp_RF = pd.read_csv("finaldata/importance_RF.csv")
df_imp_c_RF = pd.read_csv("finaldata/complete_importance_RF.csv")

df_imp_SVM = pd.read_csv("finaldata/importance_SVM.csv")
df_imp_c_SVM = pd.read_csv("finaldata/complete_importance_SVM.csv")

df_imp_MLP = pd.read_csv("finaldata/importance_MLP.csv")
df_imp_c_MLP = pd.read_csv("finaldata/complete_importance_MLP.csv")

# model performance RF, SVM and MLP
df_perf_RF = pd.read_csv("finaldata/performance_RF.csv")
df_perf_SVM = pd.read_csv("finaldata/performance_SVM.csv")
df_perf_MLP = pd.read_csv("finaldata/performance_MLP.csv")

# descriptions of the features/column names
df_featureinfotext = pd.read_csv("finaldata/feature_info.csv", sep=';')

# color theme
COLORS = {
    "background": "whitesmoke",
    "model1": '#fd7e14',
    "model2": "#446e9b",
    "model3": "#8ed1cc"
}

"""
==========================================================================
Markdown Text
"""

datasource_text = dcc.Markdown(
    """
    The data set used in this study is the ESS08 data (2016) and can be retrieved [here](https://ess-search.nsd.no/en/study/f8e11f55-0c14-4ab3-abde-96d3f14d3c76)
    """
)

interact_info_text = dcc.Markdown(
    """
> ##### _**How to use the interaction buttons**_ 
> 
> Below, you can **choose a specific target** (i.e. data set and model) or 
the average over all targets. This changes the view of all four graphs. 
Next, you can also choose **by which model to sort** the graph visualizing the feature importance
and whether to **show all features** (in case a specific target is chosen in the first interaction button). 
For the performance bar graph, you can **(un)select particular metrics** to customize your view. 
Lastly, when a specific target is selected, you can choose to either view the feature descriptions (default)
or the **cluster information**.
""", style={'color':'#5c5c5c'}
)

cluster_info_text = dcc.Markdown(
    id = 'clusterinfotext',
    style={"maxHeight":"900px", "maxWidth":"800px", "overflow":"scroll"}

)

learn_text = dcc.Markdown(
    """
This study therefore focuses on providing a starting point of such guidance by attempting to determine
essential predictors of missingness in a large, social science related data set. Ideally, combining 
this study with other studies, this will result in an archive of essential predictors so that social 
science researchers – and possibly researchers of other fields – design their studies in such a way 
that MAR is (sufficiently) guaranteed and that advanced imputation techniques can be used validly.
More specifically, in this study, a multiverse analysis approach is used, where a multitude of models is 
used to individually predict the missingness of a single target feature, but doing this for a multitude 
of targets. That is, for each target, the best model is sought by (automatic) hyperparameter tuning and 
cross-validation, after which the feature importances are calculated for the best model. Moreover, to 
make the resulting feature importances more generalizable and significant, this process is applied to not one, 
but three different types of models, namely Support Vector Machine Classifiers (SVC), Random Forest Classifiers 
(RFC), and Multilayer Perceptron Classifiers (MLPC). This allows for comparison of the feature importances – 
either strengthening or nuancing the importance of the respective features. As such, the central research question
 of this study is defined as follows:
 
_To create an archive of essential predictors, how can multiverse analysis direct the process of identifying predictors of non-response?_

The aim of this dashboard is to provide clear and interactive visualizations to explore and analyze the results of this multivariate analysis.
This dashboard consists of different graphs and plots suitable for the results at hand, following the principles described by Munzner (2014) and others. 
    """
)

learn_text2 = dcc.Markdown(
    """
The first two plots shown are variations of each other: a horizontal bar plot and the standard (vertical) bar plot. This type of plots is ideal for the tasks of looking up 
and comparing values of one quantitative value attribute, 
like feature importance and model performance. Immediately below the bar plots 
visualizing feature importance and model performance a visualization of the six 
different class maps is presented. These are scatter plots that show the 
uncertainty of the model for each observation in relation to the difficulty of the observation.
There is one scatter plot for each prediction class, so two class maps per model as the task is binary 
classification. To allow for easier comprehension, in each plot two helper lines are added to indicate the 99% quantile distance from the class (i.e. extremely difficult observations) and the midpoint 
of class uncertainty (i.e. the area where the model is most uncertain).
For the class 0 maps (left), blue is wrongly assigned to class 1 with a certainty defined by the value on the y-axis. The closer to the 0.5 line, the higher the uncertainty. The difficulty of an observation is indicated by the location on the x-axis, where further to the right indicates higher difficulty. Similarly, for the class 1 maps, orange-colored plots are wrongly assigned to class 0, a y-value of 0.5 indicates high
uncertainty and a large value on the x-axis indicates high difficulty
The last graph present in the dashboard is a correlation heat map, to allow assessing the
existence of multicollinearity for the different data sets used during the processing and of the original (cleaned) data.
    """
)


footer = html.Div(
    [
        dcc.Markdown(
            """
            This dashboard is made by Lieve Göbbels as a part of her Master's thesis "Using Multiverse Analysis for Estimating Response Models: Towards an 
            Archive of Informative Features," supervised by Kyle M. Lang at Utrecht University, The Netherlands.
            """,
        style={
                        "color": "#333333",
                        "backgroundColor": "#c6d1cc",
                        "marginTop": "5px",
                        "marginBottom": "5px",
                        "paddingTop": "45px",
                        "paddingBottom": "45px",
                        "marginLeft": "0px",
                        "marginRight":"0px",
                        "textAlign": "center",
                        # "font-family": 'Droid Sans Mono',
                    },
        ),

    ],
    className="p-2 mt-5 bg-secondary text-white small",
)

def clusterinfotext(target, datadescript):
    if (datadescript == 'show cluster info' and target != 'average'):
        df_cluster_info = pd.read_csv('~/PycharmProjects/ADSthesis/clusters info/RF/RF_{}.csv'.format(target))
        df_cluster_info.rename(columns={'Unnamed: 0': 'cluster number'}, inplace=True)
        df_cluster_info.set_index(df_cluster_info.columns[0], inplace=True)

        # drop target from list of feature names
        cols = df_original.columns.drop(target)

        # replace numbers with feature names and add empty column for iteration process later
        df_cluster_info.replace(to_replace=list(range(0, len(df_original.columns) - 1)), value=cols, inplace=True)
        df_cluster_info = df_cluster_info.fillna("")
        df_cluster_info = df_cluster_info.reset_index(drop=True)
        text = df_cluster_info.to_markdown(index=False, headers=['selected target', '1st other feature',
                                                                 '2nd', '3rd',
                                                                 '4th', '5th',
                                                                 '6th','7th',
                                                                 '8th','9th',
                                                                 '10th','11th',
                                                                 '12th','13th',
                                                                 '14th','15th'])
    else:
        df_cluster_info = df_featureinfotext.reset_index(drop=True).sort_values('Feature')
        text = df_cluster_info.to_markdown(index=False)

    return text



"""
==========================================================================
Tables
"""


# feature importance tables
importance_table1 = dash_table.DataTable(
    id="importance1",
    columns=[{"id": "Feature", "name": "Feature", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_imp_c_RF.columns[1:]
    ],
    data=df_imp_c_RF.to_dict("records"),
    sort_action='native',
    page_size=10,
    style_table={"overflowX": "scroll"},
)

importance_table2 = dash_table.DataTable(
    id="importance2",
    columns=[{"id": "Feature", "name": "Feature", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_imp_c_SVM.columns
    ],
    data=df_imp_c_SVM.to_dict("records"),
    sort_action='native',
    page_size=10,
    style_table={"overflowX": "scroll"},
)

importance_table3 = dash_table.DataTable(
    id="importance3",
    columns=[{"id": "Feature", "name": "Feature", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_imp_c_MLP.columns[1:]
    ],
    data=df_imp_c_MLP.to_dict("records"),
    sort_action='native',
    page_size=10,
    style_table={"overflowX": "scroll"},
)


# performance tables
performance_table1 = dash_table.DataTable(
    id="performance1",
    columns=[{"id": "Metric", "name": "Metric", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_perf_RF.columns[1:]
    ],
    data=df_perf_RF.to_dict("records"),
    sort_action='native',
    page_size=15,
    style_table={"overflowX": "scroll"},
)

performance_table2 = dash_table.DataTable(
    id="performance2",
    columns=[{"id": "Metric", "name": "Metric", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_perf_SVM.columns[1:]
    ],
    data=df_perf_SVM.to_dict("records"),
    sort_action='native',
    page_size=15,
    style_table={"overflowX": "scroll"},
)

performance_table3 = dash_table.DataTable(
    id="performance3",
    columns=[{"id": "Metric", "name": "Metric", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format":Format(precision=4)}
        for col in df_perf_MLP.columns[1:]
    ],
    data=df_perf_MLP.to_dict("records"),
    sort_action='native',
    page_size=15,
    style_table={"overflowX": "scroll"},
)



"""
==========================================================================
Figures
"""

def make_horbarplot(target, subset, sortby, title):

# filter data based on interaction input
    if (subset == 'subset' and target != 'average'):
        data_RF = df_imp_RF.loc[:, ['Feature',target]]
        data_RF = data_RF.dropna()
        data_SVM = df_imp_SVM.loc[:, ['Feature',target]]
        data_SVM = data_SVM.dropna()
        data_MLP = df_imp_MLP.loc[:, ['Feature',target]]
        data_MLP = data_MLP.dropna()
        height = 3000

    elif subset == 'all' or target == 'average':
        data_RF = df_imp_c_RF
        data_SVM = df_imp_c_SVM
        data_MLP = df_imp_c_MLP
        height = 6000

# sort data based on interaction input
    if sortby == 'Random Forest':
        data_RF = data_RF.sort_values(target)
        sorting = data_RF['Feature']
    elif sortby == 'Support Vector Machine':
        data_SVM = data_SVM.sort_values(target)
        sorting = data_SVM['Feature']
    else:
        data_MLP = data_MLP.sort_values(target)
        sorting = data_MLP['Feature']

    fig=make_subplots(
        specs=[[{}]]
    )

# adjust layout to place x-axis labels on top and bottom
    fig.update_layout(xaxis2={'anchor':'y', 'overlaying':'x', 'side':'top'})

# make horizontal bar plots
    fig.add_trace(
            go.Bar(
                x=data_RF[target],
                y=data_RF['Feature'],
                marker={"color": COLORS["model1"]},
                # hoverinfo="none",
                orientation='h',
                name="Random forest",
            )
    )

    fig.add_trace(
        go.Bar(
            x=data_SVM[target],
            y=data_SVM['Feature'],
            marker={"color": COLORS["model2"]},
            # hoverinfo="none",
            orientation='h',
            name="SVM"
        )
    )

    fig.add_trace(
        go.Bar(
            x=data_MLP[target],
            y=data_MLP['Feature'],
            marker={"color": COLORS["model3"]},
            # hoverinfo="none",
            orientation='h',
            name="MLP"
        )
    )

# make ghost plot for x-axis adjustment (i.e. ticks & labels on top and bottom)
    fig.add_trace(
        go.Bar(
        x=None,
        y=None,
        name='ghost',
        showlegend=False
        )
    )

# update x-axis again for x-axis adjustment (i.e. ticks & labels on top and bottom)
    fig.data[3].update(xaxis='x2')

    fig.update_xaxes(range=[0, 0.9],
                     tickmode='array',
                     tickvals=np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.85]),
                     ticktext=[0, '', 0.1, '', 0.2, 0.3, 0.5, 0.85],
                     )

# adjust layout
    fig.update_layout(
        title_text=title,
        # template="none",
        title_x=0.5,
        showlegend=True,
        legend=dict(x=0.95, y=0.99),
        margin=dict(b=40, t=75, l=35, r=30, pad=50),
        height=height,
        paper_bgcolor=COLORS["background"],
        xaxis1={'anchor':'free', 'position':1, 'side':'top', 'ticks':'outside'},
        xaxis2={'anchor':'free', 'overlaying':'x1', 'side':'bottom'},
        plot_bgcolor=COLORS["background"],
        hovermode='closest',
        margin_pad=50,
        barmode='group',
        yaxis={'categoryorder':'array', 'categoryarray':sorting},
    )

    fig.update_xaxes(showgrid=True, gridwidth=1.3, gridcolor='darkgrey')

    return fig

def make_vertbarplot(target, metrics, title):

    fig = go.Figure()

# make bar plots
    fig.add_trace(
            go.Bar(
                y= df_perf_RF[target],
                x=df_perf_RF['Metric'][df_perf_RF['Metric'].isin(metrics)],
                marker={"color": COLORS["model1"]},
                # hoverinfo="none",
                name="Random forest"
            )
    )

    fig.add_trace(
        go.Bar(
            y=df_perf_SVM[target],
            x=df_perf_SVM['Metric'][df_perf_SVM['Metric'].isin(metrics)],
            marker={"color": COLORS["model2"]},
            # hoverinfo="none",
            name="SVM"
        )
    )

    fig.add_trace(
        go.Bar(
            y=df_perf_MLP[target],
            x=df_perf_MLP['Metric'][df_perf_MLP['Metric'].isin(metrics)],
            marker={"color": COLORS["model3"]},
            # hoverinfo="none",
            name="MLP"
        )
    )

# add 0.7 threshold line
    fig.add_hline(y=0.70, line_width=2.5, line_dash=None, row='all', col='all',
                  annotation_text='Decent performance threshold (0.70)',
                  annotation=dict(font_size=11, bgcolor='white', opacity=0.7),
                  line_color='slategrey',
                  annotation_position='bottom right')


# adjust layout
    fig.update_layout(
        title_text=title,
        template="none",
        title_x=0.5,
        showlegend=True,
        legend=dict(x=0.95, y=1.15),
        margin=dict(b=40, t=90, l=40, r=15),
        height=400,
        xaxis1={'anchor':'free', 'position':0, 'side':'bottom'},
        hovermode='closest',
    )

    fig.update_xaxes(showgrid=True, gridwidth=1.3)

    return fig

def make_heatmap(target, title):

# determine data based on interaction input
    if target=='average':
        data = df_original
        features = df_original.columns

    else:
        features = df_imp_RF.loc[:, ['Feature',target]].dropna()['Feature']
        data = df_original[df_original.columns.intersection(features)]

# calculate correlation
    corr = data.corr()

    fig = go.Figure()

# make heat map
    fig.add_trace(
        go.Heatmap(
            z = corr,
            x=features,
            y=features,
            colorscale=[[0, COLORS['model2']], # manual color scale to highlight high correlation values
                        [0.3, COLORS['background']],
                        [0.7, COLORS['background']],
                        [1, COLORS['model1']]],
            zmin=-1,
            zmax=1,
            name="heatmap 1",
            hoverongaps=False,
        )
    )

# adjust layout
    fig.update_layout(
        title_text=title,
        template="none",
        title_x=0.5,
        margin=dict(b=55, t=60, l=60, r=30),
        height=1100,
        width=1100,
        xaxis=dict(scaleanchor='x', constrain='domain',
                    showticklabels=True, tickangle=45,
                   tickfont=dict(size=10)),
        yaxis=dict(showticklabels=True, tickangle=45,
                   tickfont=dict(size=10)),
        hovermode='closest',
    )

    return fig

def make_classmaps(target, title):

# make subplots space
    fig = make_subplots(rows=3, cols=2,
                        x_title='Localized Farness',
                        y_title='Probability of alternative class',
                        column_widths=[300, 300],
                        row_heights=[300,300, 300],
                        subplot_titles=['Class map of RF, class 0', 'Class map of RF, class 1',
                                        'Class map of SVM, class 0', 'Class map of SVM, class 1',
                                        'Class map of MLP, class 0', 'Class map of MLP, class 1']) # 3 models, 2 classes

    fig.update_annotations(font_size=14)

# define data based on interaction input
    if target != 'average':
        classmap_data_RF = pd.read_csv(
            '~/PycharmProjects/ADSthesis/classmaps/RF/cm_RF_{}.csv'.format(target))
        classmap_0_RF = classmap_data_RF[classmap_data_RF['class'] == 0]
        classmap_1_RF = classmap_data_RF[classmap_data_RF['class'] == 1]

        classmap_data_SVM = pd.read_csv(
            '~/PycharmProjects/ADSthesis/classmaps/SVM/cm_SVM_{}.csv'.format(target))
        classmap_0_SVM = classmap_data_SVM[classmap_data_SVM['class'] == 0]
        classmap_1_SVM = classmap_data_SVM[classmap_data_SVM['class'] == 1]

        classmap_data_MLP = pd.read_csv(
            '~/PycharmProjects/ADSthesis/classmaps/MLP/cm_MLP_{}.csv'.format(target))
        classmap_0_MLP = classmap_data_MLP[classmap_data_MLP['class'] == 0]
        classmap_1_MLP = classmap_data_MLP[classmap_data_MLP['class'] == 1]

        # scatterplot 1 for first model
        fig.add_trace(
            go.Scatter(
                x = classmap_0_RF['farness'],
                y = classmap_0_RF['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_0_RF['colors']),
                name='class 0',
                legendgroup='1',
                legendgrouptitle_text="Predicted class",
                showlegend=False,
            ),
        row=1, col=1)

        # scatterplot 2 for first model
        fig.add_trace(
            go.Scatter(
                x=classmap_1_RF['farness'],
                y=classmap_1_RF['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_1_RF['colors']),
                name='class 1',
                legendgroup='1',
                showlegend=False

            ),
        row=1, col=2)

        # scatterplot 1 for second model
        fig.add_trace(
            go.Scatter(
                x=classmap_0_SVM['farness'],
                y=classmap_0_SVM['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_0_SVM['colors']),
                name='class 0',
                legendgroup='2',
                showlegend=False,

            ),
            row=2, col=1)

        # scatterplot 2 for second model
        fig.add_trace(
            go.Scatter(
                x=classmap_1_SVM['farness'],
                y=classmap_1_SVM['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_1_SVM['colors']),
                name='class 1',
                legendgroup='2',
                showlegend=False
            ),
            row=2, col=2)

        # scatterplot 1 for third model
        fig.add_trace(
            go.Scatter(
                x=classmap_0_MLP['farness'],
                y=classmap_0_MLP['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_0_MLP['colors']),
                legendgroup='3',
                name='class 0',
                showlegend=False
            ),
            row=3, col=1)

        # scatterplot 2 for third model
        fig.add_trace(
            go.Scatter(
                x=classmap_1_MLP['farness'],
                y=classmap_1_MLP['prob alternative'],
                mode='markers',
                marker=dict(color=classmap_1_MLP['colors']),
                name='class 1',
                legendgroup='3',
                showlegend=False
            ),
            row=3, col=2)

        # ghost plots for correct legend
        fig.add_trace(
            go.Scatter(
                x=np.array([-2]),
                y=np.array([-2]),
                mode='markers',
                marker=dict(color='#fd7e14'),
                name='class 0',
                legendgroup='4',
                legendgrouptitle_text="Predicted class",
            ),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=np.array([-2]),
                y=np.array([-2]),
                mode='markers',
                marker=dict(color='#446e9b'),
                name='class 1',
                legendgroup='4',
            ),
            row=1, col=1)

# adjust layout and add helper lines
        fig.update_xaxes(range=[-0.05, qfunc(1)+0.01],
                         tickmode='array',
                         tickvals=qfunc(np.array([0, 0.5, 0.75, 0.9, 0.99, 0.999, 1])),
                         ticktext=[0, 0.5, 0.75, 0.9, 0.99, 0.999, 1])

        fig.update_yaxes(range=[-0.01, 1.05],
                         tickmode='array',
                         tickvals=[0,0.25,0.5,0.75,1.0],
                         ticktext=[0,0.25,0.5,0.75,1.0])

        fig.add_vline(x=qfunc(0.99), line_dash=None, row='all', col='all',
                      annotation_text='99% quantile distance from class',
                      annotation=dict(font_size=10),
                      line_color='slategrey')

        fig.add_hline(y=0.5, line_dash=None, row='all', col='all',
                      annotation_text='midpoint class uncertainty',
                      annotation=dict(font_size=10),
                      line_color='slategrey',
                      annotation_position='bottom right')

        fig.update_layout(
            title_text=title,
            title_x=0.47,
            height=800,
            autosize=True,
            margin=dict(b=75, t=75, l=75, r=30),
            paper_bgcolor=COLORS["background"],
            legend=dict(y=0.97),
        )

# show empty square with take-action clause when 'average' is selected
    else:
        fig = go.Figure()
        fig.update_layout(
            title_x = 0.47,
            xaxis = {'visible': False},
            yaxis = {'visible': False},
            annotations = [
                {
                    'text': "Here, the class maps will be shown<br>Please select an appropriate target",
                    'xref': "paper",
                    'yref': "paper",
                    'showarrow': False,
                    'font':{'size':28}
                }
            ],
            paper_bgcolor=COLORS["background"],
            template="none",
            margin=dict(b=400, t=400, l=400, r=400)
        )

    return fig



"""
==========================================================================
Make Tabs
"""

# =======Interaction tab components

interact_info_card = dbc.Card(interact_info_text, className="mt-2")

# interaction buttons/options
interact_card = dbc.Card(
    [

        html.H4("Choose target", className="card-title", style={"color":"#5a5a5a", "textAlign":"center"}),
        dcc.Dropdown(
            id="target",
            options=df_imp_c_RF.columns,
            value='average',
            clearable=False,
        ),
        html.Br(),
        html.H4("By which model should the horizontal bar plot be sorted?",
                className="card-title", style={"color":"#5a5a5a", "textAlign":"center"}),
        dbc.RadioItems(
            id="sortby",
            options=['Random Forest', 'Support Vector Machine', 'Multi-layer Perceptron'],
            value='Random Forest',
            inputStyle={"margin-right": "20px", "textAlign": "center"},
            label_checked_style={"color": "#5c5c5c", "font-weight":"bold"},
            input_checked_style={
                "backgroundColor": "#8da99c",
                "borderColor": "#5c5c5c",
            },
        ),
        html.Hr(),
        html.H4("Show selected subset or all features?",
                className="card-title", style={"color":"#5a5a5a", "textAlign":"center"}),
        html.H6("If average is selected, the feature importances of all features will be shown",
                style={"color":"#5a5a5a", "textAlign":"center"}),
        dbc.RadioItems(
            id="subset",
            options=['all', 'subset'],
            value='subset',
            inputStyle={"margin-right": "20px", "textAlign": "center"},
            label_checked_style={"color": "#5c5c5c", "font-weight":"bold"},
            input_checked_style={
                "backgroundColor": "#8da99c",
                "borderColor": "#5c5c5c",
            },
        ),
        html.Hr(),
        html.H4("Choose metrics",
                className="card-title", style={"color":"#5a5a5a", "textAlign":"center"}),
        dbc.Checklist(
            id="metrics",
            options=df_perf_RF['Metric'].tolist(),
            value=df_perf_RF['Metric'].tolist(),
            inputStyle={"margin-right": "20px", "textAlign": "center"},
            label_checked_style={"color": "#5c5c5c", "font-weight":"bold"},
            input_checked_style={
                "backgroundColor": "#8da99c",
                "borderColor": "#5c5c5c",
            },
        ),
        html.Hr(),
        html.H4("Show cluster information of selected target or show feature information?",
                className="card-title", style={"color":"#5a5a5a", "textAlign":"center"}),
        dbc.RadioItems(
            id="datadescript",
            options=['show cluster info', 'show feature info'],
            value='show feature info',
            inputStyle={"margin-right": "20px", "textAlign": "bottom"},
            label_checked_style={"color": "#5c5c5c", "font-weight":"bold"},
            input_checked_style={
                "backgroundColor": "#8da99c",
                "borderColor": "#5c5c5c",
            },
        ),
    ],
    body=True,
    className="mt-4",
)

# interactive text with cluster info
cluster_info_card = dbc.Card([
    dbc.CardHeader("Feature and cluster information", style={"color":"#5a5a5a", "textAlign":"center"}),
    html.Div(),
    cluster_info_text
    ], className="mt-4")


# ========= Results Tab  Components

results_card1 = dbc.Card(
    [
        dbc.CardHeader("Feature importance Random Forest"),
        html.Div(importance_table1),
    ],
    className="mt-4",
)

results_card2 = dbc.Card(
    [
        dbc.CardHeader("Feature importance Support Vector Machine"),
        html.Div(importance_table2),
    ],
    className="mt-4",
)

results_card3 = dbc.Card(
    [
        dbc.CardHeader("Feature importance Multi-Layer Perceptron"),
        html.Div(importance_table3),
    ],
    className="mt-4",
)

data_source_card1 = dbc.Card(
    [
        dbc.CardHeader("Performance Random Forest"),
        html.Div(performance_table1),
    ],
    className="mt-4",
)

data_source_card2 = dbc.Card(
    [
        dbc.CardHeader("Performance Support Vector Machine"),
        html.Div(performance_table2),
    ],
    className="mt-4",
)

data_source_card3 = dbc.Card(
    [
        dbc.CardHeader("Performance Multi-Layer Perceptron"),
        html.Div(performance_table3),
    ],
    className="mt-4",
)

# ========= Information Tab  Components
info_card = dbc.Card(
    [
        dbc.CardHeader("General information about the study and the dashboard",
                       style={"color":"#5c5c5c", "font-weight":"bold"}),
        html.Div([
            dbc.CardBody(learn_text),
            html.Hr(),
            dbc.CardBody(learn_text2),
        ])

    ],
    className="mt-4",
)


# ========= Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(info_card, tab_id="tab1", label="Info"),
        dbc.Tab(
            [interact_info_text, interact_card, cluster_info_card],
            tab_id="tab-2",
            label="Interact",
            className="pb-4",
        ),
        dbc.Tab([results_card1, data_source_card1, results_card2, data_source_card2, results_card3, data_source_card3], tab_id="tab-3", label="Results"),
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)


"""
==========================================================================
Helper function for class maps
"""

qfunc = lambda x: abs(norm.ppf(x * (norm.pdf(4) - 0.5) + 0.5))


"""
===========================================================================
Main Layout
"""

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.H1(
                    children = html.B(app_title),
                    style={
                        "color": "#333333",
                        "backgroundColor": "#c6d1cc",
                        "marginTop": "15px",
                        "marginBottom": "15px",
                        "paddingTop": "45px",
                        "paddingBottom": "45px",
                        "marginLeft": "0px",
                        "marginRight":"0px",
                        "textAlign": "center",
                        # "font-family": 'Droid Sans Mono',
                    },
                ),
            ])
        ),
        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
                        dcc.Graph(id="hor_bar_plot", className="mb-2", style={"maxHeight":"400px", "overflow":"scroll"}),
                        dcc.Graph(id="vert_bar_plot", className="pb-4"),
                        dcc.Graph(id="classmaps", className="pb-4"),
                        dcc.Graph(id="heatmap", className="pb-4"),
                        html.Hr(),
                        html.H6(datasource_text, className="my-2"),
                    ],
                    width=12,
                    lg=7,
                    className="pt-4",
                ),
            ],
            className="ms-1",
        ),
        dbc.Row(dbc.Col(footer)),
        dbc.Row(dbc.Col(about.card, width="auto"), justify="center")
    ],
    fluid=True,
)


"""
==========================================================================
Callbacks
"""

# cluster text info callback
@app.callback(
    Output("clusterinfotext", "children"),
    Input("target", "value"),
    Input("datadescript", "value")
)

def update_clusterinfotext(target, datadescript):

    text = clusterinfotext(target, datadescript)
    return text

# hor bar plot callback
@app.callback(
    Output("hor_bar_plot", "figure"),
    Input("target", "value"),
    Input("subset", "value"),
    Input("sortby", "value")
)

def update_horbarplot(target, subset, sortby):

    figure = make_horbarplot(target, subset, sortby, " Feature Importance '{}'".format(target))
    return figure

# vert bar plot callback
@app.callback(
    Output("vert_bar_plot", "figure"),
    Input("target", "value"),
    Input("metrics", "value"),
)

def update_vertbarplot(target, metrics):

    figure = make_vertbarplot(target, metrics," Performance of '{}'".format(target)) #input, title
    return figure

# class maps callback
@app.callback(
    Output("classmaps", "figure"),
    Input("target", "value"),
)

def update_classmaps(target):

    figure = make_classmaps(target," Performance of '{}'".format(target))
    return figure

# heatmap callback
@app.callback(
    Output("heatmap", "figure"),
    Input("target", "value"),
)

def update_heatmap(target):

    if target=='average':
        title = 'Original Data'
    else:
        title = target

    figure = make_heatmap(target, " Correlation heat map of '{}'".format(title))
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)