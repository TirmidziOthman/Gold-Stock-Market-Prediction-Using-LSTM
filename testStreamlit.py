import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

# Scale the data
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Set up Yahoo Finance
yf.pdr_override()

# Create a Streamlit app
st.title('Gold Stock Price Prediction with LSTM')

# Stock Symbol Input
#   stock_symbol = st.text_input("Enter stock symbol (e.g., 'GOLD'):", 'GOLD')

# Date Range Input
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Fetch stock data
df = pdr.get_data_yahoo('GOLD', start=start_date, end=end_date)
st.write("Stock Data:")
st.dataframe(df , width = 800)

_='''
st.write("Gold Close Price History")
Plot Close Price History
plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
'''

fig = px.line(df['Close'],  title = "Close Price History")
fig.update_layout( xaxis_title = 'Date', yaxis_title = 'Close Price')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

###############################################################################################

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .80 ))


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

if 'clicked' not in st.session_state:
    st.session_state.clicked = False 

#load model
model = load_model('best_model.h5')
model2 = load_model('best_model2.h5')
model3 = load_model('best_model3.h5')


# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

y_true = y_true = dataset[training_data_len:, :]

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = scaled_data[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))


model_1 , model_2, model_3 = st.tabs(["Model 1","Model 2","Model 3"])

with model_1:
    ###################################################################################################
    # best combination list hyperparameter tuning

    st.subheader('Hyperparameter Tuning')
    model1_list = pd.read_csv('hyperparameter_model1.csv')
    model1_list.columns = model1_list.columns.str.replace('Unnamed: 0','combination')
    model1List = model1_list.head(3)
    highlight = lambda x: ['background: red' if x.name in [0,0] else '' for i in x]
    st.table(model1List.style.apply(highlight, axis=1))
    #st.write(model1List.info())
   

    _='''es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    history2 = model.fit(x_train, y_train, validation_split=0.3, epochs=40, batch_size=model1_list.Batch_size[0], callbacks=[es], verbose=0)

    #plot error
    fig = plt.figure(figsize=(20,7))
    fig.add_subplot(121)

    # Accuracy
    plt.plot(history2.epoch, history2.history['root_mean_squared_error'], label = "rmse")
    plt.plot(history2.epoch, history2.history['val_root_mean_squared_error'], label = "val_rmse")

    plt.title("RMSE", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()


    #Adding Subplot 1 (For Loss)
    fig.add_subplot(122)

    plt.plot(history2.epoch, history2.history['loss'], label="loss")
    plt.plot(history2.epoch, history2.history['val_loss'], label="val_loss")

    plt.title("Loss", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()

    st.pyplot(plt)'''


    # Get the models predicted price values
    predictions1 = model.predict(x_test)
    predictions1 = scaler.inverse_transform(predictions1)


    st.write("Below is the prediction Using Best Tuning - Combination : {:d} ".format(model1_list.combination[0]))

    # Get the root mean squared error (RMSE)
    rmse =  np.sqrt(mean_squared_error(y_true, predictions1))
    #st.write("RMSE : {:.2f}%".format(rmse) )

    # Calculate Mean Absolute Percentage Error (MAPE)
    def calculate_mape(y_true, y_pred):
        assert len(y_true) == len(y_pred), "Input arrays must have the same length"
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate MAPE
    mape = calculate_mape(y_true, predictions1)

    #st.write("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

    # Plot the data
    train = data[2000:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions1
    _='''# Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    st.pyplot(plt)'''

    # Plotly line chart
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

    # Add validation data and predictions
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))

    st.subheader('Model 1')
    # Layout settings
    fig.update_layout(
        title='Actual vs Prediction',
        xaxis_title='Date',
        yaxis_title='Close Price',
        autosize=True,

    )

    st.plotly_chart(fig)

    accuracy = {'RMSE': ["{:.2f}%".format(rmse)] , 'MAPE' : ["{:.2f}%".format(mape)] }
    acc_df1 = pd.DataFrame(data=accuracy)
    st.table(acc_df1 )




############################################################################################

with model_2:

    st.subheader('Hyperparameter Tuning')
    # best combination list hyperparameter tuning
    model1_list = pd.read_csv('hyperparameter_model2.csv')
    model1_list.columns = model1_list.columns.str.replace('Unnamed: 0','combination')
    model1List = model1_list.head(3)
    highlight = lambda x: ['background: red' if x.name in [0,0] else '' for i in x]
    st.table(model1List.style.apply(highlight, axis=1))

    # Get the models predicted price values
    predictions2 = model2.predict(x_test)
    predictions2 = scaler.inverse_transform(predictions2)

    st.write("Below is the rediction Using Best Tuning - Combination : {:d} ".format(model1_list.combination[0]))

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, predictions2))
    #st.write("RMSE : {:.2f}%".format(rmse) )

    # Calculate Mean Absolute Percentage Error (MAPE)
    def calculate_mape(y_true, y_pred):
        assert len(y_true) == len(y_pred), "Input arrays must have the same length"
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate MAPE
    mape = calculate_mape(y_true, predictions2)

    #st.write("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

    # Plot the data
    train = data[2000:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions2

    # Plotly line chart
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

    # Add validation data and predictions
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))

    st.subheader('Model 2')
    # Layout settings
    fig.update_layout(
        title='Actual vs Prediction',
        xaxis_title='Date',
        yaxis_title='Close Price',
        autosize=True,

    )

    st.plotly_chart(fig)

    accuracy = {'RMSE': ["{:.2f}%".format(rmse)] , 'MAPE' : ["{:.2f}%".format(mape)] }
    acc_df2 = pd.DataFrame(data=accuracy)
    st.table(acc_df2 )





with model_3:

    st.subheader('Hyperparameter Tuning')
    # best combination list hyperparameter tuning
    model1_list = pd.read_csv('hyperparameter_model3.csv')
    model1_list.columns = model1_list.columns.str.replace('Unnamed: 0','combination')
    model1List = model1_list.head(3)
    highlight = lambda x: ['background: red' if x.name in [0,0] else '' for i in x]
    st.table(model1List.style.apply(highlight, axis=1))

    st.write("Below is the rediction Using Best Tuning - Combination : {:d} ".format(model1_list.combination[0]))

    # Get the models predicted price values
    predictions3 = model3.predict(x_test)
    predictions3 = scaler.inverse_transform(predictions3)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, predictions3))
    #st.write("RMSE : {:.2f}%".format(rmse) )

    # Calculate Mean Absolute Percentage Error (MAPE)
    def calculate_mape(y_true, y_pred):
        assert len(y_true) == len(y_pred), "Input arrays must have the same length"
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate MAPE
    mape = calculate_mape(y_true, predictions3)

    #st.write("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

    # Plot the data
    train = data[2000:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions3

    # Plotly line chart
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

    # Add validation data and predictions
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))

    st.subheader('Model 3')
    # Layout settings
    fig.update_layout(
        title='Actual vs Prediction',
        xaxis_title='Date',
        yaxis_title='Close Price',
        autosize=True,

    )

    st.plotly_chart(fig)

    accuracy = {'RMSE': ["{:.2f}%".format(rmse)] , 'MAPE' : ["{:.2f}%".format(mape)] }
    acc_df3 = pd.DataFrame(data=accuracy)
    st.table(acc_df3)


################################################################################################

st.subheader(" Best Models ")
highlight1 = lambda x: ['background: green' if x.name in [1,1] else '' for i in x]
#st.table(model1List.style.apply(highlight, axis=1))
# Your list of DataFrames
dataframes_list = [
    pd.DataFrame({'RMSE': [acc_df1['RMSE'].values], 'MAPE': [acc_df1['MAPE'].values]}),
    pd.DataFrame({'RMSE': [acc_df2['RMSE'].values], 'MAPE': [acc_df2['MAPE'].values]}),
    pd.DataFrame({'RMSE': [acc_df3['RMSE'].values], 'MAPE': [acc_df3['MAPE'].values]})
]


# Concatenate the DataFrames vertically
result_model = pd.concat(dataframes_list, ignore_index=True)
result_model.index += 1
# Display the resulting DataFrame using st.table()
st.table(result_model.style.apply(highlight1, axis=1))

st.subheader(" Future Prediction from the Best Model")
# generate the input and output sequences
n_lookback = 60 # length of input sequences (lookback period)
n_forecast =  252 * 5  # 5 years assuming about 252 trading days in a year
  # length of output sequences (forecast period)

def train_forecasting_models(scaled_data, n_lookback, n_forecast):


    model_list = pd.read_csv('hyperparameter_model1.csv')
    n_neurons1, n_neurons2, n_batch_size, dropout = list(model_list.values[0][1:5])
    #st.write (list(model_list.values[0][1:5]))

    X_fcast = []
    Y_fcast = []

    for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
        X_fcast.append(scaled_data[i - n_lookback: i])
        Y_fcast.append(scaled_data[i: i + n_forecast])

    X_fcast = np.array(X_fcast)
    Y_fcast = np.array(Y_fcast)

    regressor1 = Sequential()
    regressor1.add(LSTM(units=n_neurons1, return_sequences=True, input_shape=(n_lookback,1)))
    regressor1.add(Dropout(dropout))
    regressor1.add(LSTM(units=n_neurons2, return_sequences=False))
    regressor1.add(Dropout(dropout))
    regressor1.add(Dense(units=25))
    regressor1.add(Dense(units=1, activation='linear'))
    regressor1.add(Dense(n_forecast))
    regressor1.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    #mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    fhistory2 = regressor1.fit(X_fcast, Y_fcast, validation_split=0.2, epochs=40, batch_size=n_batch_size, callbacks=[es], verbose=0)

    # generate the forecasts
    X_ = scaled_data[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = regressor1.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # Create Plotly Figure
    fig1 = go.Figure()

    # Add Subplot 1 (For RMSE)
    fig1.add_trace(go.Scatter(x=fhistory2.epoch, y=fhistory2.history['root_mean_squared_error'], mode='lines', name='rmse'))
    fig1.add_trace(go.Scatter(x=fhistory2.epoch, y=fhistory2.history['val_root_mean_squared_error'], mode='lines', name='val_rmse'))

    fig1.update_layout(
        title='RMSE',
        xaxis_title='Epochs',
        yaxis_title='RMSE',
    )

    # Add Subplot 2 (For Loss)
    fig1.add_trace(go.Scatter(x=fhistory2.epoch, y=fhistory2.history['loss'], mode='lines', name='loss'))
    fig1.add_trace(go.Scatter(x=fhistory2.epoch, y=fhistory2.history['val_loss'], mode='lines', name='val_loss'))

    fig1.update_layout(
        title='Loss',
        xaxis_title='Epochs',
        yaxis_title='Loss',
    )

    # Show the plot using Streamlit
    st.plotly_chart(fig1)

    #data zoom
    data_zoom = data[2500:]

    st.write("Gold Stock Forecast in 5 years")

    # Organize the results in a data frame
    df_past = data_zoom[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')

    # Create Plotly Figure
    fig = go.Figure()

    # Add Actual and Forecast traces
    fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual', line=dict(width=2.0)))
    fig.add_trace(go.Scatter(x=results.index, y=results['Forecast'], mode='lines', name='Forecast', line=dict(width=2.0)))

    # Layout settings
    fig.update_layout(
        title='GOLD',
        xaxis_title='Date',
        yaxis_title='Price',
    )

    # Show the plot using Streamlit
    st.plotly_chart(fig)

if st.button("Forecast"):
    train_forecasting_models(scaled_data, n_lookback, n_forecast)
