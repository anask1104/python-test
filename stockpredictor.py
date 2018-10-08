import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bokeh
import pytest
from nsepy import get_history
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from datetime import date

tcs_stock_data = get_history(symbol='TCS',
                             start=date(2015, 1, 1),
                             end=date(2015, 12, 31))

infy_stock_data = get_history(symbol='INFY',
                              start=date(2015, 1, 1),
                              end=date(2015, 12, 31))

#print(tcs_stock_data)

# Part 1.1

def MovingAverage(closing_price, start, end, step):
    moving_average = []
    temp = []

    for i in range(start, end, step):
        for j in range(0, len(closing_price) - i + 1, 1):
            temp.append(np.average(closing_price[j: j + i]))
        moving_average.append(temp)
        temp = []

    return moving_average

movavg=MovingAverage(tcs_stock_data.Close,4,64,12)



# Part 1.3

tcs_volume = tcs_stock_data.Volume.values
infy_volume = infy_stock_data.Volume.values
tcs_close=tcs_stock_data.Close.values


def VolumeShock(volume):
    volume_shock_boolean = []
    volume_shock_direction = []
    for i in range(1, len(volume)):
        rel_diff = (volume[i] - volume[i - 1]) / volume[i]
        volume_shock_boolean.append(1 if abs(rel_diff) * 100 > 10 else 0)
        volume_shock_direction.append(1 if rel_diff > 0 else 0)

    return volume_shock_boolean, volume_shock_direction


tcs_volume_shock_boolean, tcs_volume_shock_direction = VolumeShock(tcs_volume)
tcs_volume_shock_boolean, tcs_volume_shock_direction = np.asarray(tcs_volume_shock_boolean), np.asarray(
    tcs_volume_shock_direction)


def PriceShock(price):
    price_shock_boolean = []
    price_shock_direction = []
    for i in range(0, len(price) - 1):
        rel_diff = (price[i] - price[i + 1]) / price[i]
        price_shock_boolean.append(1 if abs(rel_diff) * 100 > 2 else 0)
        price_shock_direction.append(1 if rel_diff > 0 else 0)

    return price_shock_boolean, price_shock_direction


tcs_price_shock_boolean, tcs_price_shock_direction = PriceShock(tcs_stock_data.Close)
tcs_price_shock_boolean, tcs_price_shock_direction = np.asarray(tcs_price_shock_boolean), np.asarray(
    tcs_price_shock_direction)

# Creating a data structure
tcs_closing_price = tcs_stock_data.Close.values

#Part3.3(Statements were same)

def Price_black_swan(price):
    price_black_swan_boolean = []
    price_black_swan_direction = []
    for i in range(0, len(price) - 1):
        rel_diff = (price[i] - price[i + 1]) / price[i]
        price_black_swan_boolean.append(1 if abs(rel_diff) * 100 > 2 else 0)
        price_black_swan_direction.append(1 if rel_diff > 0 else 0)

    return price_black_swan_boolean, price_black_swan_direction


tcs_price_black_swan_boolean, tcs_price_black_swan_direction = Price_black_swan(tcs_stock_data.Close)
tcs_price_black_swan_boolean, tcs_price_black_swan_direction = np.asarray(tcs_price_black_swan_boolean), np.asarray(
    tcs_price_black_swan_direction)

print(tcs_price_black_swan_boolean)

#Part 3.4
# PriceShock_without_VolumeShock=[]
# i=0
# j=1
# for f in range(tcs_close)-1:
#     shock_price[i] = (tcs_close[i] - tcs_close[i + 1]) / tcs_stock_data.Close[i]
#     shock_volume[j] = (tcs_volume[j] - tcs_volume[j - 1]) / tcs_volume[j]
#     i+=1
#     j+=1
#     print(shock_volume)
#     if (shock_price* 100 <2 and shock_volume>10):
#         PriceShock_without_VolumeShock.append()
#     print(PriceShock_without_VolumeShock)



"""Visualisation"""
#Part 2.2

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file

p = figure(plot_width=800, plot_height=400)
# add a line renderer
p.line(tcs_stock_data.index,tcs_stock_data.Close ,line_width=2)
p.xaxis.axis_label = "Date"
p.yaxis.axis_label = "Close price"
show(p)

tcs_volume_1_shock=[]
tcs_volume_0_shock=[]                                                        #0->Negative

                                                                             # 1->positive
for ind,ele in enumerate(tcs_volume_shock_boolean):
    if(ele==0):
        tcs_volume_0_shock.append(tcs_stock_data.Close[ind])
    else:
        tcs_volume_1_shock.append(tcs_stock_data.Close[ind])

print(tcs_volume_0_shock)
print(tcs_volume_1_shock)

#Part 2.3

p = figure(plot_width=800, plot_height=400)
p.multi_line([tcs_stock_data.index,tcs_stock_data.index],[tcs_volume_0_shock,tcs_volume_1_shock] ,alpha=[0.8, 0.3], line_width=2,color=["red","navy"])
p.xaxis.axis_label = "Date"
p.yaxis.axis_label = "Volume Shocks"
show(p)

#Part 2.4

movavg52=[]     #Moving average of 52  weeks
for i in movavg[4]:
    movavg52.append(i)

p = figure(plot_width=800, plot_height=400)
# add a line renderer
p.line(tcs_stock_data.index,movavg52 ,line_width=2)
p.xaxis.axis_label = "Date"
p.yaxis.axis_label = "Moving Average 52 week"
show(p)



timesteps = 30

X_train = []
y_train = []
for i in range(timesteps, len(tcs_closing_price) - 1):
    X_train.append(tcs_closing_price[i - timesteps:i])
    y_train.append(tcs_closing_price[i])
X_train, y_train = np.array(X_train), np.array(y_train)



Test_data = get_history(symbol='TCS',
                             start=date(2018, 9, 8),
                             end=date(2018, 10, 8))



#Testing data
X_test = []
y_test = []
for i in range(timesteps, len(Test_data.Close.values) - 1):
    X_train.append(Test_data.Close.values[i - timesteps:i])
    y_train.append(Test_data.Close.values[i])
X_test, y_test = np.array(X_train), np.array(y_train)


"""Part 3"""
"""Modeling"""



rdg = Ridge()
rdg.fit(X_train, y_train)

y_rid_pred = rdg.predict(X_test)
print("Predictions using ridge ",y_rid_pred)



linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_linReg_pred=linreg.predict(X_test)
print("Predictions using Linear regression",y_linReg_pred)


