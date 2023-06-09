import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
import numpy as np

def main():
    # Specify the path to the CSV file
    csv_file_path = 'Universities.csv'

    variables = ['Unnamed: 0', 'University_name', 'Region', 'Founded_year', 'Motto',     
       'UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score',
       'UG_average_fees_(in_pounds)', 'PG_average_fees_(in_pounds)',
       'International_students', 'Student_satisfaction', 'Student_enrollment', 
       'Academic_staff', 'Control_type', 'Academic_Calender', 'Campus_setting',
       'Estimated_cost_of_living_per_year_(in_pounds)', 'Latitude',
       'Longitude', 'Website']
    #target = 
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    df = preprocessing(df)
    
    target = 'UG_average_fees_(in_pounds)'
    # Split the data into independent variables (X) and the dependent variable (y)
    X = df[['UK_rank', 'World_rank']]  # Replace feature1, feature2, feature3 with your actual column names
    y = df[target]  # Replace target_variable with your actual column name

    #linearRegression(X, y)
    supportVectorRegression(X, y)


def preprocessing(data):
    # categorical features to numerical
    label_encoder = LabelEncoder()
    for x in data.columns:
        col = data[x]
        if is_object_dtype(col):
            #print(x, " is object-converting")
            data[x] = label_encoder.fit_transform(data[x])

    setMissing(data)
    return data

def setMissing(df):
    df["Founded_year"] = df["Founded_year"].replace([9999], np.nan)

def linearRegression(X, y):
    # Create an instance of the LinearRegression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    print(model.score(X, y))
    # Print the coefficients and intercept
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

def supportVectorRegression(X, y):
    # Create an instance of the LinearRegression model
    model = SVR()

    # Fit the model to the data
    model.fit(X, y)

    # Get the coefficients and intercept
    print(model.score(X, y))



if __name__ == '__main__':
    main()
