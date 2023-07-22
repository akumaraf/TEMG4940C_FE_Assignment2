import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server=app.server

# Read the data from "data.csv" using pandas
data = pd.read_csv('data.csv')
feature_names = data.columns.tolist()  # Get the names of features as a list

# Function to generate the three graphs (bar, histogram, box plot)
def generate_graphs(selected_feature):
    feature_data = data[selected_feature]
    
    # Scatter plot
    scatter_plot = go.Figure()
    scatter = go.Scatter(x=feature_data, y=data['Y-Predictor'], mode='markers')
    scatter_plot.add_trace(scatter)
    scatter_plot.update_layout(title=f'Scatter Plot: {selected_feature} vs. Y-Predictor',
                               xaxis_title=selected_feature,
                               yaxis_title='Y-Predictor')

    # Histogram
    histogram = go.Figure(data=[go.Histogram(x=feature_data)])
    histogram.update_layout(title='Histogram')
    
    # Box Plot
    box_plot = go.Figure(data=[go.Box(y=feature_data)])
    box_plot.update_layout(title='Box Plot')
    
    return scatter_plot, histogram, box_plot

def plot_confusion_matrix(confusion_matrix_csv):
    # Read confusion matrix data from the CSV file
    confusion_matrix_df = pd.read_csv(confusion_matrix_csv, index_col=0)

    # Create a heatmap to visualize the confusion matrix
    heatmap = go.Figure(data=go.Heatmap(z=confusion_matrix_df.values,
                                        x=confusion_matrix_df.columns,
                                        y=confusion_matrix_df.index))
    heatmap.update_layout(title='Confusion Matrix')

    return dcc.Graph(figure=heatmap)

# Function to plot the multi-class ROC curve from the CSV file
def plot_multiclass_roc(roc_csv):
    # Read ROC curve data from the CSV file
    roc_data = pd.read_csv(roc_csv)

    # Add endpoints for the diagonal line (line of no discrimination)
    roc_curve_data = []
    for class_label in roc_data['Class'].unique():
        roc_data_class = roc_data[roc_data['Class'] == class_label]
        roc_curve_data.append(go.Scatter(x=[0, 1],
                                         y=[0, 1],
                                         mode='lines',
                                         line=dict(color='gray', dash='dash'),
                                         showlegend=False))
        roc_curve = go.Scatter(x=roc_data_class['False Positive Rate'],
                               y=roc_data_class['True Positive Rate'],
                               name=f"Class {class_label} (AUC={roc_data_class['AUC'].iloc[0]:.2f})")
        roc_curve_data.append(roc_curve)

    layout = go.Layout(title='Multi-class ROC Curve',
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'),
                       showlegend=True)

    fig = go.Figure(data=roc_curve_data, layout=layout)
    return dcc.Graph(figure=fig)

# Define the layout for the Exploratory Data Analysis tab
eda_tab_layout = html.Div([
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in feature_names],
        value=feature_names[0],  # Set the default value to the first feature
    ),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='box-plot'),
])

#Plot feature importance for tab 3
def plot_feature_importance(importance_csv):
    # Read feature importance data from the CSV file
    importance_df = pd.read_csv(importance_csv)

    # Sort the data by importance values in descending order
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Create a bar graph to visualize the feature importances
    bar_graph = go.Figure(data=[go.Bar(x=importance_df['Feature'], y=importance_df['Importance'])])
    bar_graph.update_layout(title='Feature Importance',
                            xaxis_title='Feature',
                            yaxis_title='Importance')

    return dcc.Graph(figure=bar_graph)

# Function to plot the learning curve for tab 3
def plot_learning_curve(learning_curve_csv):
    # Read learning curve data from the CSV file
    learning_curve_df = pd.read_csv(learning_curve_csv)

    # Create the learning curve plot
    learning_curve_plot = go.Figure()

    learning_curve_plot.add_trace(go.Scatter(
        x=learning_curve_df['Training Examples'],
        y=learning_curve_df['Training Score'],
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
    ))

    learning_curve_plot.add_trace(go.Scatter(
        x=learning_curve_df['Training Examples'],
        y=learning_curve_df['Cross-validation Score'],
        mode='lines+markers',
        name='Cross-validation Score',
        line=dict(color='orange'),
    ))

    learning_curve_plot.update_layout(title='Learning Curve',
                                      xaxis_title='Training Examples',
                                      yaxis_title='Accuracy Score',
                                      showlegend=True)

    return dcc.Graph(figure=learning_curve_plot)

# Function to plot Residual Curve under tab 3
def plot_residual_plot(residual_plot_csv):
    # Read residual plot data from the CSV file
    residual_plot_data = pd.read_csv(residual_plot_csv)

    # Create the residual plot
    residual_plot = go.Figure()
    residual_plot.add_trace(go.Scatter(x=residual_plot_data['Predicted Y-Predictor'],
                                       y=residual_plot_data['Residuals'],
                                       mode='markers',
                                       marker=dict(color='blue'),
                                       showlegend=False))

    residual_plot.update_layout(title='Residual Plot',
                                xaxis_title='Predicted Y-Predictor',
                                yaxis_title='Residuals',
                                showlegend=False)

    return dcc.Graph(figure=residual_plot)

# Callback to update the graphs based on the selected feature
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('histogram', 'figure'),
     Output('box-plot', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_graphs(selected_feature):
    scatter_plot, histogram, box_plot = generate_graphs(selected_feature)
    return scatter_plot, histogram, box_plot

# Define the layout for the Models' Evaluation tab
models_evaluation_tab_layout = html.Div([
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Gradient Boosting Classifier', 'value': 'gradient_boosting'},
            {'label': 'Random Forest Classifier', 'value': 'random_forest'},
            {'label': 'Decision Tree Classifier', 'value': 'decision_tree'},
        ],
        value='random_forest'  # Default value for dropdown
    ),
        dbc.Row([
        dbc.Col(dbc.Card(id='accuracy-card', body=True, color='info', outline=True), width=3),
        dbc.Col(dbc.Card(id='recall-card', body=True, color='info', outline=True), width=3),
        dbc.Col(dbc.Card(id='precision-card', body=True, color='info', outline=True), width=3),
        dbc.Col(dbc.Card(id='f1-score-card', body=True, color='info', outline=True), width=3),
    ], style={'margin-top': '20px'}),  # Add margin between the row of cards and the search bar
    # Add the confusion matrix and multi-class ROC curve plots side by side
    dbc.Row([
        dbc.Col(html.Div(id='confusion-matrix-plot'), width=6),
        dbc.Col(html.Div(id='multiclass-roc-plot'), width=6),
    ]),
])

# Define the layout for the Model's Insight Generation tab
insight_generation_tab_layout = html.Div([
    dcc.Dropdown(
        id='plot-dropdown',
        options=[
            {'label': 'Feature Importance', 'value': 'feature_importance'},
            {'label': 'Learning Curve', 'value': 'learning_curve'},
            {'label': 'Residual Plot', 'value': 'residual_plot'},
        ],
        value='feature_importance'  # Default value for dropdown
    ),
    # Add a div to display the selected plot
    html.Div(id='selected-plot'),
])

# Callback to update the selected plot for tab 3 based on the dropdown value
@app.callback(
    Output('selected-plot', 'children'),
    [Input('plot-dropdown', 'value')]
)
def update_selected_plot(selected_plot):
    if selected_plot == 'feature_importance':
        return plot_feature_importance('feature_importance.csv')
    elif selected_plot == 'learning_curve':
        return plot_learning_curve('learning_curve_data.csv')
    elif selected_plot == 'residual_plot':
        return plot_residual_plot('residual_plot_data.csv')
    else:
        return html.Div("Select a plot from the dropdown")


# Define the main layout of the app using tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploratory Data Analysis', value='tab-1', children=eda_tab_layout),
        dcc.Tab(label="Models' Evaluation", value='tab-2', children=models_evaluation_tab_layout),
        dcc.Tab(label="Decision Tree Model's Insight Generation", value='tab-3', children=insight_generation_tab_layout),
    ], style={'margin-bottom': '20px'}),  # Add margin between the tabs and the dropdowns
])


# Define a callback to update the content of the "Model's Evaluation" tab based on the dropdown selection
@app.callback(
    [Output('confusion-matrix-plot', 'children'),
     Output('multiclass-roc-plot', 'children'),
     Output('accuracy-card', 'children'),
     Output('recall-card', 'children'),
     Output('precision-card', 'children'),
     Output('f1-score-card', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_model_evaluation_content(selected_model):
    confusion_matrix_csv = None
    roc_curve_csv = None
    evaluation_metrics_csv = None

    if selected_model == 'gradient_boosting':
        confusion_matrix_csv = 'gbc_confusion_matrix.csv'
        roc_curve_csv = 'gbc_roc_curve.csv'
        evaluation_metrics_csv = 'gbc_evaluation_metrics.csv'
    elif selected_model == 'random_forest':
        confusion_matrix_csv = 'rfc_confusion_matrix.csv'
        roc_curve_csv = 'rfc_roc_curve.csv'
        evaluation_metrics_csv = 'rfc_evaluation_metrics.csv'
    elif selected_model == 'decision_tree':
        confusion_matrix_csv = 'dt_confusion_matrix.csv'
        roc_curve_csv = 'dt_roc_curve.csv'
        evaluation_metrics_csv = 'dt_evaluation_metrics.csv'

    confusion_matrix_plot = plot_confusion_matrix(confusion_matrix_csv)
    roc_curve_plot = plot_multiclass_roc(roc_curve_csv)

    # Read evaluation metrics data from the CSV file
    metrics_data = pd.read_csv(evaluation_metrics_csv)

    accuracy_card = dbc.CardBody([
        html.H4('Accuracy', className='card-title'),
        html.P(f"{metrics_data['Value'][0]:.5f}", className='card-text')
    ])

    recall_card = dbc.CardBody([
        html.H4('Recall', className='card-title'),
        html.P(f"{metrics_data['Value'][1]:.5f}", className='card-text')
    ])

    precision_card = dbc.CardBody([
        html.H4('Precision', className='card-title'),
        html.P(f"{metrics_data['Value'][2]:.5f}", className='card-text')
    ])

    f1_score_card = dbc.CardBody([
        html.H4('F1-Score', className='card-title'),
        html.P(f"{metrics_data['Value'][3]:.5f}", className='card-text')
    ])

    return confusion_matrix_plot, roc_curve_plot, accuracy_card, recall_card, precision_card, f1_score_card



if __name__ == '__main__':
    app.run_server(debug=True)
