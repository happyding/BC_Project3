# Initial imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

  # Importing required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%matplotlib inline

def read_symbol_file(symbol_file):
    symbol_df = pd.read_csv(symbol_file)

    symbol_df = symbol_df.rename(columns={'Unnamed: 0': 'DataDate'})
    symbol_df = symbol_df.set_index(pd.to_datetime(symbol_df.DataDate, infer_datetime_format=True))
    symbol_df = symbol_df.drop(symbol_df.columns[0], axis=1)
    
    return symbol_df

def process_lstm_symbol_file(symbol_df):

    #Create new trading signals Df, Set index as datetime object and drop extraneous columns
    trading_signals_df = pd.DataFrame()

    #add daily change rates to increase the staionarity of dataset
    trading_signals_df['volume delta'] = symbol_df['Volume'].dropna().pct_change()
    trading_signals_df['bb std delta'] = symbol_df['bollinger_std'].dropna().pct_change()
    trading_signals_df['rvol delta'] = symbol_df['rvol'].dropna().pct_change()
    trading_signals_df['option rvol delta'] = symbol_df['Option rVol'].dropna().pct_change()

    #add daily returns as target
    trading_signals_df['daily returns'] = symbol_df['daily returns'].dropna()

    trading_signals_df= trading_signals_df.fillna(value = 0)
    trading_signals_df= trading_signals_df.replace([np.inf, -np.inf], 0.0)
    trading_signals_df
    
    return trading_signals_df

def get_window_data(symbol_signals_df, window_size, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(symbol_signals_df) - window_size):
        features = symbol_signals_df.iloc[i : (i + window_size), feature_col_number]
        
        #print(features)
            
        target = symbol_signals_df.iloc[(i + window_size), target_col_number]
        
        
        X.append(features)
        y.append(target)
        
    return np.array(X), np.array(y).reshape(-1, 1)

def lstm_split_scale_reshape(X_df, y_df, split_ratio):
    split_range = int(split_ratio * len(X_df))

    X_train = X_df[: split_range]
    X_test = X_df[split_range:]

    y_train = y_df[: split_range]
    y_test = y_df[split_range:]
    
    
    scaler = MinMaxScaler()
    scaler.fit(X_df)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    scaler.fit(y_df)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test, scaler

def run_lstm(number_units, dropout_fraction, epochs, X_train, X_test, y_train):
    
    # Define the LSTM RNN model.
    model = Sequential()

    # Layer 1
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1),
        activation = 'relu')
    ) 
    model.add(Dropout(dropout_fraction))
    
    # Layer 2
    model.add(LSTM(units=number_units, return_sequences=True, activation = 'relu'))
    model.add(Dropout(dropout_fraction))
    
    # Layer 3
    model.add(LSTM(units=number_units, activation = 'relu'))
    model.add(Dropout(dropout_fraction))
    
    # Output layer
    #model.add(Dense(1))
    model.add(Dense(1, activation = 'linear'))
    
    # Compile the LSTM model
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Show the model summary
    print(model.summary())
    
    # Train the model
    print(model.fit(X_train, y_train, epochs=epochs, shuffle=False, batch_size=1, verbose=1))
    
    # Make predictions using the testing data X_test
    predicted_df = model.predict(X_test)
    
    return predicted_df

def process_predicted_data(predicted_df, y_test, scaler, symbol_signals_df):
    
    #Recover the original, nonscaled prices
    predicted_prices = scaler.inverse_transform(predicted_df)
    real_prices = scaler.inverse_transform(y_test.reshape(-1,1)) 
    
    shift =7
    stocks = pd.DataFrame({
        "Actual": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index = symbol_signals_df.index[-len(real_prices)-shift:-shift ] )
    
    return stocks

def process_stock_data(stocks, shift):
    
    #convert stocks df into positive and negative signals
    stocks['Positive Actual signal'] = np.where(stocks['Actual'] > 0, 1, 0)
    stocks['Negative Actual signal'] = np.where(stocks['Actual'] < 0, -1, 0)

    stocks['Positive Predicted signal'] = np.where(stocks['Predicted'] > 0, 1, 0)
    stocks['Negative Predicted signal'] = np.where(stocks['Predicted'] < 0, -1, 0)

    #merge to create one column per signal, shifted back to reflect forward projection window
    
    stocks['Actual Signal'] = stocks['Positive Actual signal'] + stocks['Negative Actual signal']
    stocks['LSTM Predicted Signal'] = stocks['Positive Predicted signal'] + stocks['Negative Predicted signal']
    
    return stocks

def main_lstm(symbol_file):

    symbol_df = read_symbol_file(symbol_file)

    symbol_signals_df = process_lstm_symbol_file(symbol_df)
    
    window_size = 7
    feature_col_number = 1 
    target_col_number = -1
    X, y = get_window_data(symbol_signals_df, window_size, feature_col_number, target_col_number)
    
    split_ratio = 0.7
    X_train, X_test, y_train, y_test, scaler = lstm_split_scale_reshape(X, y, split_ratio)
    
    number_units = 50
    dropout_fraction = 0.2
    epochs = 50
    predicted = run_lstm(number_units, dropout_fraction, epochs, X_train, X_test, y_train)

    stocks_df = process_predicted_data(predicted, y_test, scaler, symbol_signals_df)

    shift = 7
    stocks_df = process_stock_data(stocks_df, shift)

    return stocks_df

# symbol_file = "etf_MTUM_initial.csv"

# result = main_lstm(symbol_file)
# result.head(30)



# #set path to Features CSV and read in CSV
# symbol_file = "TSLA_results.csv"

# symbol_df = read_symbol_file(symbol_file)

# #symbol_df.info()

# symbol_signals_df = process_lstm_symbol_file(symbol_df)

# print(symbol_signals_df.head(15))

# window_size = 7
# feature_col_number = 1 
# target_col_number = -1

# # Create the features (X) and target (y) data using the window_data() function.
# X, y = get_window_data(symbol_signals_df, window_size, feature_col_number, target_col_number)
# print(X[:15])
# print(y[:15])

# split_ratio = 0.7
# X_train, X_test, y_train, y_test, scaler = lstm_split_scale_reshape(X, y, split_ratio)


# number_units = 50
# dropout_fraction = 0.2
# epochs = 50

# predicted = run_lstm(number_units, dropout_fraction, epochs, X_train, X_test, y_train)
# predicted

# stocks_df = process_predicted_data(predicted, y_test, scaler, symbol_signals_df)

# shift = 7

# stocks_df = process_stock_data(stocks_df, shift)
    
# stocks_df





#stocks['LSTM Predicted Signal'].to_pickle(r'tsla_LSTM.pickle')

#stocks.plot(y = ['Actual Signal', 'LSTM Predicted Signal'], figsize = (20,10), kind = 'bar')

# Calculate cumulative return of model and plot the result
#(1 + (stocks['Actual'] * stocks['LSTM Predicted Signal'])).cumprod().plot(figsize = (20,10))

# Set initial capital allocation
#initial_capital = 100000

# Plot cumulative return of model in terms of capital
#cumulative_return_capital = initial_capital * (1 + (stocks['Actual'] * stocks['LSTM Predicted Signal'])).cumprod()
#cumulative_return_capital.plot(figsize = (20,10))

