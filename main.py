import pandas as pd
import glob
import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import sklearn
## reading sepsis and no_sepsis datasets
## I choose 1,5,7,8,14,21,30,31,39,42 th csv in sepsis folder, And 2,21,22,24,31,39,42,43,44,45 th csv in no-sepsis folder
path = r'sepsis'
path2 = r'no_sepsis'
filenames_sepsis = glob.glob(path+'/*.csv') # glob.glob saves a path
filenames_no_sepsis = glob.glob(path2+'/*csv')
dataframes = []
for filename in filenames_sepsis:
    dataframes.append(pd.read_csv(filename)) # I am saving all dataframes in a list
for filename in filenames_no_sepsis:
    dataframes.append(pd.read_csv(filename))

# concatenating csv files
concatenatedFrame = pd.concat(dataframes,ignore_index=True) # pd.concat concatenates dataframes, it can take dataframes as a list.
# dropping unused columns from dataframe
# df.drop returns df after drop given columns if parameter axis = 1 and if parameter inplace = true df will change in memory.
concatenatedFrame.drop(concatenatedFrame.columns[[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]],axis=1,inplace=True)
# df.median gives medians of columns and df.fillna fills all nan values with given value
concatenatedFrame = concatenatedFrame.fillna(concatenatedFrame.median())

## df.max() gives maximum values of columns and df.min() gives minimum values of columns
max_heart_rate = concatenatedFrame.max()[0] # 161
min_heart_rate = concatenatedFrame.min()[0] # 45
med_heart_rate = concatenatedFrame.median()[0] # 87.5
max_temp = concatenatedFrame.max()[1] # 39.94
min_temp = concatenatedFrame.min()[1] # 3.77
med_temp = concatenatedFrame.median()[1] # 37.113

# creating Antecedent (input/sensor) variables for a fuzzy control system.
# ctrl.Antecedent parameters are universe(array-type) , label(string)
heart_rate_antecedent = ctrl.Antecedent(np.arange(min_heart_rate,max_heart_rate,1),'heart_rate')
temp_antecedent = ctrl.Antecedent(np.arange(min_temp,max_temp,1),'temp')

#fuzz.trimf is a Triangular membership function generator.
# It takes a 1d array and another 1d array like[a,b,c] with 3 length which edges of triangular a<=b<=c.
# a and c are x value when y 0,  and b is x value when y 1


#fuzz.trapmf is a Trapezoidal membership function generator.
# It takes a 1d array and another 1d array with 4 length which edges of Trapezoidal which edges of trapezoidal a<=b<=c<=d.
# a and d are x value when y 0 , and b and c are x value when y 1
heart_rate_antecedent['low'] = fuzz.trimf(heart_rate_antecedent.universe,[min_heart_rate,min_heart_rate,med_heart_rate])
heart_rate_antecedent['low-medium'] = fuzz.trimf(heart_rate_antecedent.universe,[min_heart_rate, (min_heart_rate+med_heart_rate)/2,med_heart_rate])
heart_rate_antecedent['high-medium'] = fuzz.trimf(heart_rate_antecedent.universe,[65, med_heart_rate,130])
heart_rate_antecedent['high'] = fuzz.trapmf(heart_rate_antecedent.universe,[med_heart_rate,135,max_heart_rate,max_heart_rate])
temp_antecedent['low'] = fuzz.trapmf(temp_antecedent.universe,[min_temp,min_temp,33,med_temp])
temp_antecedent['low-medium'] = fuzz.trimf(temp_antecedent.universe,[33,35,med_temp])
temp_antecedent['high-medium'] = fuzz.trimf(temp_antecedent.universe,[35,med_temp,max_temp])
temp_antecedent['high'] = fuzz.trimf(temp_antecedent.universe,[med_temp,max_temp,max_temp])

# creating Consequent (output/control) variable for a fuzzy control system.
# ctrl.Consequent parameters are universe(array-type) , label(string)

sepsis_consequent = ctrl.Consequent(np.arange(0,2,1),'sepsis')
sepsis_consequent['no'] = fuzz.trapmf(sepsis_consequent.universe,[0,0,0.4,0.6])
sepsis_consequent['yes'] = fuzz.trapmf(sepsis_consequent.universe,[0.4,0.6,1,1])
'''
number of datasets according to universes.

low low sepsis : 0 no sepsis : 0
low low-medium sepsis : 0 no sepsis : 33
low high-medium sepsis : 0 no sepsis : 25
low high sepsis : 0 no sepsis : 0
low-medium low sepsis : 0 no sepsis : 0
low-medium low-medium sepsis : 41 no sepsis : 69
low-medium high-medium sepsis : 30 no sepsis : 37
low-medium high sepsis : 0 no sepsis : 3
high-medium low sepsis : 2 no sepsis : 0
high-medium low-medium sepsis : 28 no sepsis : 33
high-medium high-medium sepsis : 58 no sepsis : 12
high-medium high sepsis : 8 no sepsis : 0
high low sepsis : 0 no sepsis : 0
high low-medium sepsis : 17 no sepsis : 9
high high-medium sepsis : 49 no sepsis : 16
high high sepsis : 10 no sepsis : 0

'''

'''
loop for checking numbers of given data which in a specific universe.
For instance, number of patiences sepsis and non-sepsis with low heart rate and medium high temp 

ctyes = 0
ctno = 0
for i in range(concatenatedFrame.shape[0]):
    if concatenatedFrame['heart_rate'][i] >= 110 and concatenatedFrame['temp'][i] < 38.45 and concatenatedFrame['temp'][i] >= 37.1:
        if concatenatedFrame['sepsis_icd'][i] == 1:
            ctyes +=1
        else:
            ctno +=1
print(ctyes,ctno)
'''

### rules defined according majority of in every universes.
rule1 = ctrl.Rule(heart_rate_antecedent['low'] & temp_antecedent['low'],sepsis_consequent['no'])
rule2 = ctrl.Rule(heart_rate_antecedent['low'] & temp_antecedent['low-medium'],sepsis_consequent['no'])
rule3 = ctrl.Rule(heart_rate_antecedent['low'] & temp_antecedent['high-medium'],sepsis_consequent['no'])
rule4 = ctrl.Rule(heart_rate_antecedent['low'] & temp_antecedent['high'],sepsis_consequent['no'])
rule5 = ctrl.Rule(heart_rate_antecedent['low-medium'] & temp_antecedent['low'],sepsis_consequent['no'])
rule6 = ctrl.Rule(heart_rate_antecedent['low-medium'] & temp_antecedent['low-medium'],sepsis_consequent['no'])
rule7 = ctrl.Rule(heart_rate_antecedent['low-medium'] & temp_antecedent['high-medium'],sepsis_consequent['no'])
rule8 = ctrl.Rule(heart_rate_antecedent['low-medium'] & temp_antecedent['high'],sepsis_consequent['no'])
rule9 = ctrl.Rule(heart_rate_antecedent['high-medium'] & temp_antecedent['low'],sepsis_consequent['yes'])
rule10 = ctrl.Rule(heart_rate_antecedent['high-medium'] & temp_antecedent['low-medium'],sepsis_consequent['no'])
rule11 = ctrl.Rule(heart_rate_antecedent['high-medium'] & temp_antecedent['high-medium'],sepsis_consequent['yes'])
rule12 = ctrl.Rule(heart_rate_antecedent['high-medium'] & temp_antecedent['high'],sepsis_consequent['yes'])
rule13 = ctrl.Rule(heart_rate_antecedent['high'] & temp_antecedent['low'],sepsis_consequent['yes'])
rule14 = ctrl.Rule(heart_rate_antecedent['high'] & temp_antecedent['low-medium'],sepsis_consequent['yes'])
rule15 = ctrl.Rule(heart_rate_antecedent['high'] & temp_antecedent['high-medium'],sepsis_consequent['yes'])
rule16 = ctrl.Rule(heart_rate_antecedent['high'] & temp_antecedent['high'],sepsis_consequent['yes'])

# ctrl.ControlSystem is a base class to contain a Fuzzy Control System.
# It takes rules as parameter.
sepsis_control = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,
                                     rule10,rule11,rule12,rule13,rule14,rule15,rule16])

# ctrl.ControlSystemSimulation calculates results from a ControlSystem.
# It takes control system as parameter.
sepsis_simulation = ctrl.ControlSystemSimulation(sepsis_control)

# predicts list saves prediction of our fuzzy systems
# if result > 0.5 it assumes the patience sepsis. Otherwise, it assumes the patieence non-sepsis.
predicts = []
for i in range(concatenatedFrame.shape[0]):
    sepsis_simulation.input['heart_rate'] = concatenatedFrame['heart_rate'][i]
    sepsis_simulation.input['temp'] = concatenatedFrame['temp'][i]
    sepsis_simulation.compute()
    if sepsis_simulation.output['sepsis'] >= 0.5:
        predicts.append(1.0)
    else:
        predicts.append(0.0)
# correct_values saves result of sepsis value in used datasets.
correct_values = []
for i in range(concatenatedFrame.shape[0]):
    correct_values.append(concatenatedFrame['sepsis_icd'][i])

# confusion_matrix computes confusion matrix to evaluate the accuracy of a classification.
# parameters are ground truth (correct) target values and estimated targets as returned by a classifier.
# returns confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
confusion_mat = confusion_matrix(correct_values,predicts)
print(confusion_mat)
TP , FP , FN , TN = confusion_mat[0,0],confusion_mat[0,1],confusion_mat[1,0],confusion_mat[1,1]
TPR = TP/(TP+FN)
FPR = FP/(FP+TN)
TNR = TN/(FP+TN)
FNR = FN / (TP+FN)

f1_score = (TP / (TP+0.5*(FP+FN) ) )

print(TPR ,FPR , TNR, FNR , f1_score)

print(roc_auc_score(correct_values,predicts))


# Set the title and labels of the plot
# plt.title('Heart Rate Antecedent')
# plt.xlabel('Heart Rate (bpm)')
# plt.ylabel('Fuzzy Membership')
'''
# Change the colors of the membership functions
# plotting heart_rate_antecedent
heart_rate_antecedent['low'].view(sim=sepsis_simulation, fill='red')
heart_rate_antecedent['low-medium'].view(sim=sepsis_simulation, fill='orange')
heart_rate_antecedent['high-medium'].view(sim=sepsis_simulation, fill='yellow')
heart_rate_antecedent['high'].view(sim=sepsis_simulation, fill='green')
'''
'''
# Change the colors of the membership functions
# plotting temp_antecedent
temp_antecedent['low'].view(sim=sepsis_simulation, fill='red')
temp_antecedent['low-medium'].view(sim=sepsis_simulation, fill='orange')
temp_antecedent['high-medium'].view(sim=sepsis_simulation, fill='yellow')
temp_antecedent['high'].view(sim=sepsis_simulation, fill='green')
'''
# sepsis_consequent['no'].view(sim=sepsis_simulation, fill='yellow')
# sepsis_consequent['yes'].view(sim=sepsis_simulation, fill='green')

# True positive rates and false positive rates
fpr, tpr, _ = roc_curve(correct_values,  predicts)

# Plot the ROC curve
plt.plot(fpr, tpr)

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# Show the plot
plt.show()







