import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import csv
import matplotlib.pyplot as plt
import datetime
import math
import statistics
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy
import datetime
import csv



def convert_file(filename):
        s = 0
        df = pd.read_csv(filename)
        l = pd.to_datetime(df.date)
        m = []
        day = []
        seasoner = []
        for j in l:
                s = s+1
                day.append(s)
        q = df.power
        mydict = {'hour':day,'power':q}
        data = pd.DataFrame(mydict,columns =['hour','power'])
        return data,df





# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
#series = pd.read_csv('rt.csv', header=None)
# seasonal difference

def create_plot(data,og_data,z):
#print(math.sqrt(mean_squared_error(np.array(data.power[0:1401]).reshape(-1,1),np.array(predictions).reshape(-1,1))))
    pyplot.plot(og_data.power[0:z])
    pyplot.plot(data.power[0:z], color='red')
    pyplot.show()


def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

def predict_days(no_of_days,filename,z):
    data,df_2 = convert_file(filename)



    minuter1 = []
    minuter = []
    X = data.power
    size = len(X)-7
    train, test = X[0:size], X[size:]
    history = [x for x in train]
    predictions = list()
    for t in test:
	    model = ARIMA(history, order=(1,1,1))
	    model_fit = model.fit(trend='nc', disp=False)
	    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	    resid = model_fit.resid
	    diff = difference(history)
	    yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
	    predictions.append(yhat)
	#obs = test[t]
	    history.append(yhat)
	    print('>predicted=%.3f, actual=%.3f' % (yhat, t))
    rmse = sqrt(mean_squared_error(test, predictions))

    print('Test RMSE: %.3f' % rmse)

    print(len(predictions))
    for i in range(len(predictions)):
        minuter1.append(i)
    mydicter1 = {'hour':minuter1,'power':predictions}
    datar1 = pd.DataFrame(mydicter1,columns = ['hour','power'])

    minuter2 = []
    print(len(test))
    for i in range(len(test)):
        minuter2.append(i)
    mydicter2 = {'hour':minuter2,'power':test}
    datar2 = pd.DataFrame(mydicter2,columns = ['hour','power'])

    predicter_2 = []
    forecast = model_fit.forecast(no_of_days*1440)
    y = forecast[1]
    print(y)
    for j in y:
        predicter_2.append(j)


    for i in range(len(y)):
        minuter.append(i)
    mydicter = {'hour':minuter,'power':predicter_2}
    datar = pd.DataFrame(mydicter,columns = ['hour','power'])
    data_1 = convert_minutes_to_days(datar,no_of_days,z,df_2)
        #create_plot(data,datar1,z)
    #create_plot(datar1,datar2,z)
    return data_1



def convert_minutes_to_days(data,no_of_days,t,df):
        r = 0
        t_hours = t
        print(type(t_hours))
        z = 0
        finale = []
        hourer = []
        while r < len(data.hour):
          sim = data.power[r:r+t_hours]
          finale.append(np.sum(sim))
          r = r + t_hours
          z = z + 1
          hourer.append(z)
        new_dates = give_date(df,len(hourer))
        mydicter = {'hour':new_dates,'power':finale}
        datar = pd.DataFrame(mydicter,columns = ['hour','power'])
        return datar[0:no_of_days]

#def convert_to_date_from_utc(data):


def give_date(df_og,no_of_days):
        format_str = '%Y-%m-%d'
        z = df_og.date[len(df_og)-1]
        date = datetime.datetime.strptime(z,format_str)
        s = []
        for i in range(no_of_days):
                date += datetime.timedelta(days=1)
                s.append(date)
        return s
#pyplot.plot(tes)
print(data.power[998])
