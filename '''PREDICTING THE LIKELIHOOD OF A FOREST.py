'''PREDICTING THE LIKELIHOOD OF A FOREST FIRE'''




#-------------------------------imports-------------------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


#------------------------------importing dataset---------------------
dataset = pd.read_csv('ForestFireDataset.csv')


#-----------------------------cleaning data-----------------------------
X = dataset.drop(columns = ['Fire Occurrence', 'Area'])
y = dataset['Fire Occurrence']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


#----------------------------user inputs-------------------
Oxy_Value = int(input("Provide the Oxygen value: "))
Temp_value = int(input("Provide the temperature value(in celcius): "))
humidity_Value = int(input("Provide the humidity value: "))
user_inputset = [Oxy_Value,Temp_value,humidity_Value]


#--------------- training the model -----------------------
model  = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict([user_inputset])


#------------------Output---------------------------------
if predictions == 1:
    print ("Probability for Forest fire is very high in this area.")
elif predictions == 0:
    print("Probability for forest fire is very low in this area.")









