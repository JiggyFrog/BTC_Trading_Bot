import json
import random
import hmac
import requests
import Model
import time
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta

import DatasetInserter

Classifier = Model.classifierModel('ClassifierModel.keras', 1, 2)
getBTC = 'https://api.xeggex.com/api/v2/market/candles'
import hashlib
try:
    from urllib import urlencode
except:
    from urllib.parse import urlencode

# This is where the Private key is loaded from, the file doesn't exist because that is my private API key
superSecretKey = open('superSecretKey.txt', 'r').readline()

def getNow():
    """Gets the BTC -> USD price data"""
    f = datetime.datetime.now(datetime.timezone.utc) - timedelta(minutes=15*7)
    t = datetime.datetime.now(datetime.timezone.utc)
    f = int(f.timestamp()*1000)
    t = int(t.timestamp()*1000)
    parameters = {
        'symbol': 'BTC/USDT',
        'from': f,
        'to': t,
        'resolution': '15',
        'countBack': 6
    }
    print(parameters)
    q = requests.get(getBTC, params=parameters).json()
    return q


def hmac_sha256(key, payload):
    """Generates an HMAC-SHA256 signature for the given message using the provided key."""
    key = key.encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def convertToX(js):
    """Converts the json received from the Xeggex API into what the model uses for an input"""
    if len(js)==6:
        print("VALID at", datetime.datetime.now().isoformat())
        x = []
        closes = []
        for bar in js:
            Close, Open, High, Low, Volume = bar['close'], bar['open'], bar['high'], bar['low'], bar['volume'] * bar[
                'close']
            closes.append(Close)
            x.append([Open, High, Low, Close, Volume])

        emas: list = calculate_ema(closes, 5)
        for i in range(6):
            x[i].append(emas[i])
        return np.array([x])
    else:
        raise IndexError


def createHeader(url, param):
    """Creates a header for Xeggex's HMAC authentication"""
    nonce = str(int(time.time() * 1000))
    signature = hmac_sha256(superSecretKey, '1728040ff43a41e51a4f1c20a225ff48'+url+param+nonce)
    return {
        "X-API-KEY": '1728040ff43a41e51a4f1c20a225ff48',
        "X-API-NONCE": nonce,
        "X-API-SIGN": signature
    }



def calculate_ema(data, span):
    """Calculates the Exponential Moving Average (EMA) for a given list."""

    df = pd.DataFrame({'data': data})
    return df['data'].ewm(span=span, adjust=False).mean().tolist()


def getUSDTBal():
    r = requests.get('https://api.xeggex.com/api/v2/balances',
                     headers=createHeader('https://api.xeggex.com/api/v2/balances', '')).json()
    output = []
    for cur in r:
        if cur['asset'] == 'USDT':
            output.append(cur)
        elif cur['asset'] == 'BTC':
            output.append(cur)
    return output


def toBTC(x):
    """Gets the price of BTC"""
    res = requests.get('https://api.xeggex.com/api/v2/pool/getbysymbol/BTC_USDT').json()
    print(res['lastPrice'])
    return x/float(res['lastPrice'])


def makeTransaction(amt,sellAmt, buySell):
    """Makes a transaction using the Xeggex API"""
    bal = getUSDTBal()
    print(bal)
    if buySell == 'buy':
        amt = str(toBTC(float(bal[1]["available"])) * amt)
    else:
        amt = str(float(bal[0]["available"]) * sellAmt)
    print(buySell, amt)
    orderType = 'market'
    symbol = 'BTC_USDT'
    url = 'https://api.xeggex.com/api/v2/createorder'
    params = {
        "symbol": symbol,
        "side": buySell,
        "type": orderType,
        "quantity": amt,
    }
    data_str = json.dumps(params, separators=(',',':'))
    header = createHeader(url, data_str)
    r = requests.post(url, data=params, headers=header)
    z = r.status_code
    print(r.text)
    if z == 401:
        print("Auth Error")
    if z == 429:
        print("Rate Limit")
    if z == 400:
        print("Broke Ahh Feller")
    return z

# Loads a dataset
ds = DatasetInserter.dataSet("loading.json", "btcData.csv", debug=True, save=True)


# PARAMETERS
# These are the hyperparameters for the trading part of the program
buyThreshold = 0.78
sellThreshold = 0.55
betAmount = 0.9  # Percentage of balance to buy with
sellAmt = 1  # Percentage of balance to sell with
ongoingTransaction = False  # Manually has to be set because I don't think Xeggex API lets you see if you have an active trade
sidelineMode = True  # If Sideline mode is False then it will make real trades

# PARAMETERS

previous = None
previous_prediction = None
total = 0
right = 0
while True:
    total += 1
    print("-"*20, "New Scan", "-"*20)
    now = getNow()
    x = convertToX(now['bars'])
    cur_predicition = Classifier(x)[0][0]
    print("Prediction:", cur_predicition)

    # Re-fit the model with new data if it got it wrong
    if previous and previous_prediction:
        if previous[0]['close'] < now['bars'][-1]['close']:
            right = 1
        else:
            right = 0
        if previous_prediction > 0.5:
            previous_prediction = 1
        else:
            previous_prediction = 0
        if not previous_prediction == right:
            x, y = ds.getDataSlice(np.array(convertToX(previous)), np.array([right]), size=12000)
            Classifier.fitModelLoop(x, y, epochs=16)
            print("Trained!")
        else:
            right += 1
            print("Model Was Correct!")
    print("Model Accuracy:", right/total*100, "%")

    # making the Trade
    if not sidelineMode:
        if cur_predicition > buyThreshold and not ongoingTransaction:
            makeTransaction(betAmount, sellAmt, 'buy')
            ongoingTransaction = True
        elif cur_predicition < sellThreshold and ongoingTransaction:
            makeTransaction(betAmount, sellAmt, 'sell')
            ongoingTransaction = False
        else:
            print("Holding Steady!")
    else:
        if cur_predicition > buyThreshold and not ongoingTransaction:
            print("Would have bought!", cur_predicition)
            ongoingTransaction = True
        elif cur_predicition < sellThreshold and ongoingTransaction:
            print("Would Have Sold!", cur_predicition)
            ongoingTransaction = False
        else:
            print("Holding Steady!")

    previous_prediction = cur_predicition
    previous = now['bars']
    time.sleep(14*60)
