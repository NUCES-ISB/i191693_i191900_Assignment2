from twelvedata import TDClient
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Initialize client - apikey parameter is required
td = TDClient(apikey="26d290a4b47748589898a6791d779474")

# Construct the necessary time series
ts = td.time_series(
    symbol="AAPL",
    interval="1min",
    outputsize=10,
    timezone="America/New_York",
)

# Returns pandas.DataFrame
df = ts.as_pandas()

# Extract the predictor and target variables
X = df.index.values.reshape(-1, 1)
X = pd.DataFrame((df.index.astype(int) / 10**9).astype("int32"))
y = df["close"].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LinearRegression object
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Create the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    style={
        "max-width": "800px",
        "margin": "0 auto",
        "font-family": "Arial, sans-serif",
    },
    children=[
        html.H1(
            children="Linear Regression Results",
            style={"text-align": "center", "margin-bottom": "50px"},
        ),
        html.P(
            children=f"R^2 score: {r2:.2f}",
            style={"font-size": "24px", "font-weight": "bold", "margin-bottom": "30px"},
        ),
        dcc.Graph(
            id="future-prices",
            figure={
                "data": [
                    {
                        "x": df.index,
                        "y": df["close"],
                        "type": "scatter",
                        "name": "Actual",
                        "marker": {"color": "blue"},
                    },
                    {
                        "x": df.index,
                        "y": model.predict(X),
                        "type": "scatter",
                        "name": "Predicted",
                        "marker": {"color": "red"},
                    },
                ],
                "layout": {
                    "title": {
                        "text": "Predicted vs Actual Prices",
                        "font": {"size": 28},
                    },
                    "xaxis": {"title": {"text": "Time"}},
                    "yaxis": {"title": {"text": "Price"}},
                    "legend": {"font": {"size": 18}},
                },
            },
            style={"height": "500px", "margin-bottom": "50px"},
        ),
        html.P(
            children="Data source: Twelve Data API",
            style={"font-size": "14px", "font-style": "italic", "text-align": "center"},
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
