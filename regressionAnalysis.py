import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#I worked with Aaron and he walked me through how to get the simpleanalysis part of the problem set
#I used mostly stackoverflow and found videos online



class AnalysisData:
    
    def _init_(self,filename):
        self.variables = []
        self.filename = filename
    
    def parseFile(self):
        self.dataset = pd.read_csv(self.filename)
        self.variables = self.dataset.columns

parseData = AnalysisData('./candy-data.csv')
parseData.parseFile()



class LinearAnalysis:
    
    def _init_(self,targetY):
        
        self.bestX = ""
        self.targetY = targetY
        self.fit = ""
    
    
    def runSimpleAnalysis(self,parseData):
        
            dataset = parseData.dataset
            
            best_predictor = 0
            for column in parseData.variables:
                if column == self.targetY or column == 'competitorname":
                    continue
                
                x_values = parseData[column].values.reshape(-1,1)
                y_values = parseData[self.targetY].values
                
                regress = LinearRegression()
                regress.fit(x_values, y_values)
                predictors = regress.predict(x_values)
                points = r2_score(y_values, predictors)
                
                if points > best_predictor:
                    best_predictor = points
                    self.bestX = column
    
        self.fit = best_predictor
        print(self.bestX)
        print(self.fit)

linear_analysis = LinearAnalysis(targetY = 'sugarpercent')
linear_analysis.runSimpleAnalysis(parseData)






#class LogisticAnalysis:
#variables become what type of variable is

#def_init_(self,targetY)
# self.bestX = ""
# self.targetY = Ytarget


#functions do linear regressions here


