import pandas as pd
import numpy as np
import numpy

hot=50
csv_for_count =pd.read_csv("output.csv").values
csv_=open('output.csv')
l = (len(csv_.readlines())-5)  ##l is count of rows in csv
dt=np.arange(l)
week_list=np.arange(65520).reshape(52,7,12,15)
count_1=np.arange(1260).reshape(7,12,15)
count_1=np.zeros((7,12,15))

#list_csv=np.arange(65520*4).reshape(65520,4)
list_csv_b=np.arange(65520).reshape(65520,1)

a2=np.arange(1260,dtype=float).reshape(7,12,15)
a2=np.zeros((7,12,15))



r=0
for b in range(0,52):
    for c in range(0,7):
        for d in range(0,12):
            for e in range(0,15):
                week_list[b][c][d][e]=r
print('meke list values 0 --- Done')

print('Counting ..........')   
x=0    
for b in range(0,l):
    for c in range(0,52):
        if(csv_for_count[b,0]==c):
            for d in range(0,7):
                 for e in range(0,12):
                     if(csv_for_count[b,1]==d and csv_for_count[b,2]==e):
                         for f in range(0,15):
                             if(csv_for_count[b,3]==f):
                                 week_list[c][d][e][f]=week_list[c][d][e][f].astype(int)+1
##                                 print(c)
                                 x=x+1
##                                 print(b," ",c," ",d," ",e," ",f," "," ",week_list[c][d][e][f])

print('Counting Completed........')

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
values = array(data)
#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                 

cols = 3
rows = 65520
list_csv = np.zeros((rows, cols, 15)) 
##matrix_b[0, 0] = np.array([0, 0, 1])
#matrix_b[0, 0] = [0, 0, 1]


                       
print('creating data array..........')                                         
x=np.arange(65520)                   
b=0
cont=-1
tot=0
#enoding and creating feature set
for a in range(0,52):
    for b in range(0,7):
        for c in range(0,12):
            for d in range(0,15):
                cont=cont+1
                list_csv[cont][0]=onehot_encoded[b]
                list_csv[cont][1]=onehot_encoded[c]
                list_csv[cont][2]=onehot_encoded[d]
                tot=tot+week_list[a][b][c][d]
                #list_csv[cont][3]=week_list[a][b][c][d]

#data
cont=-1
for a in range(0,52):
    for b in range(0,7):
        for c in range(0,12):
            for d in range(0,15):
                cont=cont+1
                list_csv_b[cont][0]=week_list[a][b][c][d]


#print(list_csv[43530][:])
#print(list_csv_b[43530])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data=list_csv[:][0:3][0]

print('jgjhgjgjgjghjgjhgjh')
target=list_csv_b
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

from sklearn.linear_model import LinearRegression

clsfr=LinearRegression()
clsfr.fit(train_data,train_target)
results=clsfr.predict(test_data)

from sklearn.metrics import r2_score

r2_value=r2_score(test_target,results)
print('r2_score:',r2_value)
print('Actual value:',test_target[0:10])
print('Predicted value',results[0:10])
##
##
##pickle.dump(data,open('data.pickle','wb'))
##pickle.dump(target,open('target_count.pickle','wb'))
##print('pickle file creating ---- Done')






#print(list_csv)
#my_df = pd.DataFrame(list_csv)
#my_df.to_csv('my_csv_with_count.csv', index=False, header=False)
#print("csv to dataset -- done ")
#print(tot)
#import creat_dataset_3                    
                
                
                

                
              























                                   

##print(week_list)   
    
