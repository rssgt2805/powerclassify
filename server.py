import socket
import json

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

import pickle

from sklearn.externals import joblib


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
        return data,df,day





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
def predict_days(no_of_days,filename,z):
        data,df_2,h = convert_file(filename)
        X = data.power
        #print(len(X))
        no_of_minutes = no_of_days*z;
        minutes_in_year = round(0.66*len(X))
        differenced = difference(X, minutes_in_year)
        # fit model
        '''
        model = ARIMA(differenced, order=(1,1,1))
        print(len(differenced))
        model_fit = model.fit(disp=0)
        joblib.dump(model_fit, 'filename.pkl')
        '''
        model_fit = joblib.load('filename.pkl')
        # multi-step out-of-sample forecast
        start_index = len(differenced)
        end_index = start_index + no_of_minutes
        forecast = model_fit.predict(start=start_index, end=end_index)
        
        print(forecast)
        # invert the differenced forecast to something usable
        history = [x for x in X]
        day = 1
        predictions = []
        minuter = []
        for yhat in forecast:
                inverted = inverse_difference(history, yhat, minutes_in_year)
                #print('minute %d: %f' % (day, abs(inverted)))
                predictions.append(inverted)
                history.append(inverted)
                day += 1
                
        for i in range(len(predictions)):
                    minuter.append(i)   
        mydicter = {'hour':minuter,'power':predictions}
        datar = pd.DataFrame(mydicter,columns = ['hour','power'])
        data_1 = convert_minutes_to_days(datar,no_of_days,z,df_2)
        plt.plot(h,df_2.power,color = 'red' ,label = 'regression line')
        '''
        plt.plot(h,datar.power[0:len(h)],color = 'blue', label = 'scatter plot')
        plt.xlabel('hour')
        plt.ylabel('power')
        plt.legend()
        plt.show()
        print(mean_squared_error(df_2.power,datar.power[0:len(df_2.power)]))
        print(df_2.power,datar.power[0:len(df_2.power)])
        '''
        return data_1['power'].values.tolist()
                

def convert_minutes_to_days(data,no_of_days,y,df):
        r = 0
        t_hours = y
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
#print(data.power[998])
s = socket.socket()
port = 1234
s.bind(('localhost', port))
s.listen(5)
c, addr = s.accept()
print ("Socket Up and running with a connection from",addr)
while True:
    rcvdData = json.loads(c.recv(1024).decode())

    print ("S:",rcvdData,type(rcvdData))

    try:
        

        if(int(rcvdData.get("days")) == 1 or int(rcvdData.get("days")) == 2):
            
            print("Prediction Model ")
            
            if(int(rcvdData.get("days")) == 1):
                a = 7
            else:
                a = 30
            out = predict_days(a,'data_reg_final_4.csv',1440)
            
            print("Prediction Model 2")
            
            result = { "type" : "prediction" , "data" : out  }
            
            print(result)
            
            sendData = json.dumps(result)
            c.send(sendData.encode())

    except Exception as e:
        print(e)

        if(rcvdData == "edited"):
            sendData = "OK"
            c.send(sendData.encode())    
        else:    
            print("else")
            sendData = "Edited"
            c.send(sendData.encode())

        if(rcvdData == "Bye"):
            break
    # if(sendData == "Bye" or sendData == "bye"):
    #     break
c.close()