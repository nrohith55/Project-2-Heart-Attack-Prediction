# Project Title : Heart-Attack-Prediction

Business Objective : The main aim of the project is to check whether a person will get heart attack or not?

Following are the steps followed during any data science projects:(Project Flow):

1)Define Goal (Business Objective) : Understanding the business objective.What exactly we are going to predict here.We should know all feature details either from domain knowledge oe from your experience.In this project the main aim of the project is to find whether a person will get heart attack or not?

2)Data collection and getting Data set details: Once we get any data set during the project ,we should know the meaning of each and every single feature in it.Following are the steps involved in getting the data set details: a)Find the number columns ,number of rows. b)What is training and testing data set ratio? c)Find the dimension of the data. d)Know what all different data types in the data set.

3)EDA :(Exploratory Data Analysis)

Here we are exploring and vizualizing data.Try to understand the pattern in your data.Here we try to understand relationship of features with the outcome.EDA step is very important part in any type of project.EDA is noting but Exploratory data analysis.Here we are getting insights from the data.Exploratory Data Analysis (EDA) helps in understanding the data sets by summarizing their main characteristics and plotting them visually.This step is very important before we apply and Machine learning algorithm.Following are the steps involved: a)We should know about Business moments such as Measure of Central tendency(Mean,Median.Mode),Measure of dispersion(Variance,Standatd deviation,Range),Skewness,Kurtosis. b)Check whether the data is normally disrtibuted or not.If it is not Normally distributed apply scaling technique. c)Graphical representation :Represent the data set with the help of Graphs such as Line plot,Bar plot,Histogram plot,Pie chart,Box plot and try to get inferences/insights from it. d)we should find the correlation in the data set.All the features should be independent to each other.

4)Data Preprocessing:

Data preprocessing is noting but cleaning the data and removing unwanted material out of your data like noise,duplicate records,inconsistancy,missing values.Following are the steps involved a)Data Cleaning step plays a very important role here.In this project we have both categorical and numeric data .While building ML models we have to give numeric inputs so categorical values should be converted to mathematical form before model building.So we use dummy method for creating those values.

5)Model Building step:
First step in Model building is partitioning the data.Here we alredy have train and test data set seprately. In our project we are dealing with classification probelm i.e o find whether the given drug has a side effect or not with the help of reviews.Following are the models used for Model building in this project: 1)Logistic regression. 2)Decision tree classifier. 3) Random forest classifier. 4) Extra tree classifier . 5) Support Vector Machine(SVM) Classifier.(Linear SVM) 6)Neural Networks 7)Bagging classifier method 
How to solve Class Imbalance problem in data set? By using SMOTE technique we are trying to solve class imbalance problem.

6)Evalute and Compare Performance :

In this step we finalised the models based on accuracy ,precison,recall and f1 score values.We get all these values from Confusion Matrix.We are checking here the performance of the classification models.And also we check for the errors in the different models.We are comparing the actual value versus predicted values.Also we are checking cochens kappa score.

7)Model Deployment and draw insights :

Here we finalized Logistic regression model since it has better accuracy ,precison,recall and f1 score values than all other models.Also we can observe less errors and cochens kappa score is 0.72 which is substantial. For deployment we have create two files like app.py and Model.pkl file is also created.

Flask File creation: ● In the second file named server.py in this we have import flask ● Use @app.route(‘/’) url to execute homepage function which named as DOCTYPE.html ● Use @app.route(‘/predict’, mothods=[‘post’]) transfer data to server and provide input after that we can see the prediction using this link http://127.0.0.1:5000/

These are the above steps are followed during this Project
