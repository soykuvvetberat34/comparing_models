import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\R2 values\\maaslar.csv")
edu=datas.iloc[:,1].values.reshape(-1,1)
salary=datas.iloc[:,2].values.reshape(-1,1)
#Linear Regression model
Linear_regressor=LinearRegression()
Linear_regressor.fit(edu,salary)
predict_LinearR=Linear_regressor.predict(edu)
R2_score_LinearR=r2_score(salary,predict_LinearR)

#Polynomial regression model
Poly_Reg=PolynomialFeatures(degree=4)
degreed_values=Poly_Reg.fit_transform(edu)
degreed_LRegressor=LinearRegression()
degreed_LRegressor.fit(degreed_values,salary)
predict_polynomial=degreed_LRegressor.predict(degreed_values)
R2_score_Poly=r2_score(salary,predict_polynomial)

#Support Vector Regression model
standard_scaler_edu = StandardScaler()
edu_scaled = standard_scaler_edu.fit_transform(edu)
standard_scaler_salary = StandardScaler()
salary_scaled = standard_scaler_salary.fit_transform(salary).ravel()
svr_reg = SVR(kernel="rbf")
svr_reg.fit(edu_scaled, salary_scaled)
predict_svr = svr_reg.predict(edu_scaled)
R2_score_SVR = r2_score(salary_scaled, predict_svr)

#Decision Tree regression model(overfitting)
DT_reg=DecisionTreeRegressor()
DT_reg.fit(edu,salary)
predict_dt=DT_reg.predict(edu)
R2_score_DT=r2_score(salary,predict_dt)

#RandomForest regression model
RF_reg=RandomForestRegressor(n_estimators=10,random_state=0)
RF_reg.fit(edu,salary)
predict_RF=RF_reg.predict(edu)
R2_score_RF=r2_score(salary,predict_RF)

print("R2 Score for Linear Regression:", R2_score_LinearR)
print("R2 Score for Polynomial Regression:", R2_score_Poly)
print("R2 Score for Support Vector Regression:", R2_score_SVR)
print("R2 Score for Decision Tree Regression:", R2_score_DT)
print("R2 Score for Random Forest Regression:", R2_score_RF)

















