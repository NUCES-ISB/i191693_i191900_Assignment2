import dash
from dash import dcc
from dash import html
import pandas as pd
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import yfinance as yf
import base64
from io import BytesIO
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Define the layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='TSLA Stock Price Prediction Dashboard'),
    
    dcc.Graph(
        id='future-prices-plot'
    ),

    html.Div(children=[
        html.H2(children='Linear Regression Results'),
        html.Table([
            html.Tr([html.Th('Future Prices'), html.Td(id='future-prices')]),
            html.Tr([html.Th('Future Dates'), html.Td(id='future-dates')]),
            html.Tr([html.Th('R^2 Score'), html.Td(id='r2-score')]),
        ]),
    ])
])

# Define a Dash callback to update the dashboard when the page loads
@app.callback(
    [dash.dependencies.Output('future-prices-plot', 'figure'),
     dash.dependencies.Output('future-prices', 'children'),
     dash.dependencies.Output('future-dates', 'children'),
     dash.dependencies.Output('r2-score', 'children')],
    [dash.dependencies.Input('future-prices-plot', 'id')]
)
def update_dashboard(_):
    # Download TSLA stock data from Yahoo Finance
    data = yf.download("TSLA", start="2020-01-01", end="2023-03-22")

    # Create a new DataFrame with the closing prices of TSLA
    df = pd.DataFrame(data['Close'])

    # Define the training data and labels
    X_train = df[:-30].values
    y_train = df[30:].values

    # Create a new linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the future prices of TSLA
    future_prices = model.predict(df[-30:].values)
    future_dates = df[-30:].index.strftime('%Y-%m-%d')

    # Calculate the R^2 score of the model
    r2 = r2_score(y_train, model.predict(X_train))

    # Create a line plot of the predicted future prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Future Prices'))
    fig.update_layout(title='Predicted Future Prices of TSLA', xaxis_title='Date', yaxis_title='Price ($)')

    # Save the plot to a bytes object
    buffer = BytesIO()
    fig.write_image(buffer, format='png')
    buffer.seek(0)

    # Convert the bytes object to a base64 encoded string
    image_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the linear regression results and plot as outputs to update the dashboard
    return fig, str(list(future_prices)), str(list(future_dates)), str(r2)

if __name__ == '_main_':
    app.run(debug=False, host = "0.0.0.0", port= 8000)