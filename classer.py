
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
import socket
import re
import matplotlib.pyplot as plt

# In[6]:



def convert_input_file(filename):
        s = 0 
        df = pd.read_csv(filename)
        df.columns = ['date','power']
        l = pd.to_datetime(df.date)
        m = []
        day = []
        for j in l:
                s = s+1    
                day.append(s)
        q = df.power
        mydict = {'hour':day,'power':q}
        data = pd.DataFrame(mydict,columns =['hour','power'])
        return data,df,day


z,n,g = convert_input_file(r'demo_data.csv')
#print(z)


# In[3]:


def convert_train_file(filename):
    df = pd.read_csv(filename)
    j = df.columns[1:]
    j = list(j)
    mydict = {'ref':df[j[0]],'kit_1':df[j[1]],'kit_2':df[j[2]],'micro':df[j[3]],'light':df[j[4]]}
    data = pd.DataFrame(mydict,columns =['ref','kit_1','kit_2','micro','light'])
    return df,data,j
    
df,data_1,j = convert_train_file(r'labels.csv')    


# In[4]:


#print(j)
#t = input()
df = df.fillna(df.mean())
#print(df)
target_data = []
target = []
for i in j:
    for k in df[i]:
        target_data.append(k)

#target_data = np.array(target_data).reshape(-1, 1)

for i in j:
    for k in df[i]:
        target.append(i)
#print(target_data)        
#print(target)   

Encoder = LabelEncoder()
Y = Encoder.fit_transform(target)
#print('hello ml')
#print(Y)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(target_data,Y,test_size=0.1)





# 4,0,1,3,2

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim = 1,activation='relu'))
    model.add(Dense(10, input_dim = 1, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))
    #model.add(Dense(4, input_dim=2, activation='relu'))
    #model.add(Dense(4, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.fit(Train_X,Train_Y, epochs=2, batch_size=4, verbose=0)
    return model



#t = input()
X_check = []
#X_check.append(int(t))
filename = 'filename.h5'
loaded_model = joblib.load(open(filename, 'rb'))
result_score = loaded_model.score(Test_X,Test_Y)
#result = loaded_model.predict(X_check)
#print(result_score)


# In[53]:


def lister_it(lister,threshold):        
        #lister = lister[0:100] 
        print(lister)
        filename = 'filename.h5'
        loaded_model = joblib.load(open(filename, 'rb'))
        result_score = loaded_model.score(Test_X,Test_Y)
        my_dict = {4:'0' , 0:'0' , 1:'0' ,2:'0' , 3:'0'} 
        time_start = {4:0 , 0:0 , 1:0 ,2:0 , 3:0} 
        time_stop = {4:0 , 0:0 ,1:0 , 2:0 , 3:0}
        tot_time = {4:0 , 0:0 ,1:0 , 2:0 , 3:0} 
        checker = [] 
        checker.append(lister[0]) 
        #res_intial = loaded_model.predict(checker) 
        #my_dict[int(res_intial)] = '1' 
        #print(res_intial) 
        #print(my_dict)
        for i in range(len(lister) - 1):
            diff = lister[i+1] - lister[i] 
            #print(diff) 
            if diff > 0 and diff > threshold: 
                check = [] 
                check.append(int(diff)) 
                res = loaded_model.predict(check)
                print('#',res)
                #print(res)
                my_dict[int(res)] = '1'
                time_start[int(res)] = i
                print(dicter_1[int(res)] + 'on')
                #print(my_dict)
        
            elif (diff * -1) < threshold and (diff*-1)> 0 :
                continue

            elif diff > 0 and diff < threshold:
                    continue
            else:
                check_1 = []
                diff = int(diff)
                check_1.append(diff * -1)
                res = loaded_model.predict(check_1)
                #print(res)
                time_stop[int(res)] = i
                tot = time_start[int(res)] - time_stop[int(res)]
                if tot < 0:
                        tot = tot * -1
                        tot_time[int(res)] = tot_time[int(res)] + tot 
                my_dict[int(res)] = '0'
                print(dicter_1[int(res)] + 'off')
        #print(my_dict)
        #print(my_dict,tot_time)
        return my_dict,tot_time

dicter_1 = {4:'refrigerator',0:'kitchen_outlet_1',1:'kitchen_outlet_2',3:'microven',2:'lighting'}
df = pd.DataFrame(dicter_1, columns = ['refrigerator','kitchen_outlet_1','kitchen_outlet_2','microven','lighting'])


threshold = input()
content_buffer = [] 
s = socket.socket()         
s.bind(('0.0.0.0', 8090 ))
s.listen(0)                 
 
while len(content_buffer) < int(threshold):
 
    client, addr = s.accept()
 
    while True:
        content = client.recv(32)
 
        if len(content) ==0:
           break
        elif len(content_buffer) > int(threshold) :
            break;
        else:
            digit = re.findall(r'\d+',content.decode("utf-8"))
            content_buffer.append(int(digit[0]))
            appliance,time_used = lister_it(content_buffer,20)
            
            print(appliance,time_used)      
            print(content)

print(content_buffer) 
print("Closing connection")
appliance,time_used = lister_it(content_buffer,10)
print(appliance,time_used)
x = []
for key, value in time_used.items():
      x.append(value)
print(x)      
plt.hist(x,density=True, bins=5,alpha = 0.5)
plt.show()

client.close()
    
