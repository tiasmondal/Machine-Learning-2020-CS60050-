import numpy as np
import csv
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import pprint

filename = "data_modified_Decision_tree.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
# outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
# temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
# humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
# windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
# play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
# dataset ={'outlook':outlook,'temp':temp,'humidity':humidity,'windy':windy,'quality':play}
# df = pd.DataFrame(dataset,columns=['outlook','temp','humidity','windy','quality'])
# data=df;
print(data);
print(data["quality"].value_counts())
def find_entropy(data):
	entropy_node = 0  #Initialize Entropy
	values = data["quality"].unique()  #Unique objects - 'Yes', 'No'
	for value in values:
		fraction = data["quality"].value_counts()[value]/len(data["quality"])  
		entropy_node += -fraction*log(fraction)
	return(entropy_node)


def ent(data,attribute):
	target_variables = data["quality"].unique()  #This gives all 'Yes' and 'No'
	variables = data[attribute].unique()    #This gives different features in that attribute (like 'Sweet')


	entropy_attribute = 0
	for variable in variables:
		entropy_each_feature = 0
		for target_variable in target_variables:
			num = len(data[attribute][data[attribute]==variable][data["quality"] ==target_variable]) #numerator
			den = len(data[attribute][data[attribute]==variable])  #denominator
			fraction = num/(den+eps)  #pi
			entropy_each_feature += -fraction*log(fraction+eps) #This calculates entropy for one feature like 'Sweet'
		fraction2 = den/len(data)
		entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy ETaste

	return(abs(entropy_attribute))
headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"];
#headers=['outlook','temp','humidity','windy']
print("Entropy at root level")
for i in range(0,len(headers)):
	print(headers[i]+" "+str(ent(data,headers[i])));

def find_winner(data):
	Entropy_att = []
	IG = []
	for key in data.keys()[:-1]:
#           Entropy_att.append(find_entropy_attribute(df,key))
		IG.append(find_entropy(data)-ent(data,key))
	return IG


def get_subtable(data, node,value):
  return data[data[node] == value].reset_index(drop=True);
x=np.zeros(11);
print("Priority nodes")
def buildTree(data,x,tree=None): 
	Class = data.keys()[-1]   #To make the code generic, changing target variable class name
	
	
	#Here we build our decision tree
	# x[0]=x[0]+1
	# if(x[0]>=25):                #Max iteration limit
	# 	return tree;
	#Get attribute with maximum information gain
	IG = find_winner(data)
	node=data.keys()[:-1][np.argmax(IG)]
	IGvalue=IG[np.argmax(IG)]
	print(node)
	#Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
	attValue = np.unique(data[node])
	
	#Create an empty dictionary to create tree    
	if tree is None:                    
		tree={}
		tree[node] = {}
	
   #We make loop to construct a tree by calling this function recursively. 
	#In this we check if the subset is pure and stops if it is pure. 

	for value in attValue:
		
		subtable = get_subtable(data,node,value)
		clValue,counts = np.unique(subtable['quality'],return_counts=True)                        
		#print(counts)
		#print(clValue)
		if len(counts)==1 or IGvalue<=10e-15:#Checking purity of subset
			if(len(counts)==1):
				tree[node][value] = clValue[0]
			if(IGvalue<=10e-15):
				tree[node][value] = clValue[np.argmax(counts)]
				

		else:
			if(len(subtable["quality"])<=10):
				tree[node][value] = clValue[np.argmax(counts)]
				return tree;
			tree[node][value] = buildTree(subtable,x) #Calling the function recursively

				   
	return tree

x=[0];
tree=buildTree(data,x)
print(tree)
pprint.pprint(tree);

##################### Testing Decision Tree #######################
# for j in range(0,8):
# 	i=0;
# 	while(i<len(headers)):
# 		try:
			
# 			#print("i ="+str(i));
			
# 			tree=tree[headers[i]][data[headers[i]][j]];
# 			print(tree);
# 			if(tree=='yes' or tree == 'no'):
# 				print(tree);
				
# 			else:
# 				i=-1;
# 			i=i+1;

# 		except:
# 			i=i+1;


