import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def main():
    # Specify the path to the CSV file
    csv_file_path = 'Universities.csv'

    features = ['University_name', 'Region', 'Founded_year', 'Motto',
       'UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score',
       'International_students', 'Student_satisfaction', 'Student_enrollment',
       'Academic_staff', 'Control_type', 'Academic_Calender', 'Campus_setting',
       'Estimated_cost_of_living_per_year_(in_pounds)', 'Latitude',
       'Longitude', 'Website']
    #target =
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    nan_count = df["Motto"].isna().sum()
    print(nan_count )

    df = preprocessing(df)

    target1 = 'UG_average_fees_(in_pounds)'
    target2 = 'PG_average_fees_(in_pounds)'

    target = 'UG_average_fees_(in_pounds)'
    # plotting
    # plotting(df, target)
    # plot_test(df)
    #plotting(df, target1)


    # Split the data into independent variables (X) and the dependent variable (y)
    X = df[['UK_rank', 'World_rank']]  # Replace feature1, feature2, feature3 with your actual column names
    y = df[target1]  # Replace target_variable with your actual column name

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42)
    #linearRegression(X, y)
    #supportVectorRegression(X_train, X_test, y_train, y_test)
    #random_forest_regression(X_train, X_test, y_train, y_test)
    #supportVectorRegression(X, y)
    multiLayerPerceptron(X_train, X_test, y_train, y_test)

def plot_contourplot(data, var1, var2):
    fig2 = sns.kdeplot(x=data[var1], y=data[var2], legend=True)

    plt.title('{} - {}'.format(var1, var2))
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.savefig('plots/contour-{}-{}-1.png'.format(var1, var2))
    #plt.show()
    plt.close()

def plot_test(data):
    g = sns.PairGrid(data)
    g.map(sns.scatterplot)
    plt.show()

def plot_relationship(data, var1, var2):
    # Create scatter plot of two variables using Matplotlib
    plt.scatter(data[var1], data[var2])
    plt.title('{} - {}'.format(var1, var2))
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.savefig('plots/scatter-{}-{}-1.png'.format(var1, var2))
    plt.close()
    #plt.show()

    plot_contourplot(data, var1, var2)


def plotting(data, target):
    for x in data.columns:
        if x != target:
            plot_relationship(data, x, target)

def preprocessing(data):
    # categorical features to numerical
    label_encoder = LabelEncoder()
    mappings = {}
    for x in data.columns:
        col = data[x]
        if is_object_dtype(col):
            # print(x, " is object-converting")
            col[pd.isnull(col)] = 'NaN'
            data[x] = col
            data[x] = label_encoder.fit_transform(data[x])

            le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            if 'NaN' in le_name_mapping:
                data[x] = data[x].replace([le_name_mapping['NaN']], np.nan)

            mappings[x] = le_name_mapping
            #print(le_name_mapping)

    setMissing(data)
    return data

def setMissing(df):
    df["Founded_year"] = df["Founded_year"].replace([9999], np.nan)

def linearRegression( X_train, X_test, y_train, y_test):
    print("Linear Regression: ")


def multiLayerPerceptron(X_train, X_test, y_train, y_test):
    params = {
        "solver": ["lbfgs", "adam"],
        "learning_rate_init": [0.005,0.005, 0.01, 0.1],
        "alpha": [1e-4, 1e-5, 1e-6, 1e-7]
    }
    model = MLPRegressor(solver='lbfgs', alpha=1e-5,learning_rate_init=0.001, hidden_layer_sizes=(10,), random_state=1, max_iter=1000)
    cv = GridSearchCV(model, params)
    cv.fit(X_train, y_train)

    print(cv.score(X_train, y_train))
    print(cv.score(X_test, y_test))

def linearRegression(X_train, X_test, y_train, y_test):
    # Create an instance of the LinearRegression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X_train, y_train)

    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    print(model.score(X_test, y_test))
    # Print the coefficients and intercept
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

def supportVectorRegression( X_train, X_test, y_train, y_test):

    rfr = SVR()
    params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "C": [0.01, 0.1, 1.0, 3.],
        "epsilon": [0.01, 0.1, 1]
    }
    cv = GridSearchCV(rfr, params)
    cv.fit(X_train, y_train)

    print("Support Vector Regression:")
    # Create an instance of the LinearRegression model
    model = SVR(kernel="linear")

    # Fit the model to the data
    model.fit(X_train, y_train)

    # Get the coefficients and intercept
    print(model.score(X_test, y_test))


def random_forest_regression(X_train, X_test, y_train, y_test):
    PLOT_SCORE_OVER_N_EST = True
    
    print(f"Using {X_train.shape[1]} features for random forest regression")

    scores = list()
    
    num_estimators = np.arange(10, 200, 5)    
    
    for num_estimator in num_estimators:
        rf_regressor = RandomForestRegressor(n_estimators=num_estimator)
        rf_regressor.fit(X_train, y_train)
        score = rf_regressor.score(X_test, y_test)
        #print(f"Num Estimators = {num_estimator}, Score = {score}")
        scores.append(score)
    
    if PLOT_SCORE_OVER_N_EST:
        fig, ax = plt.subplots()
        fig.set_size_inches((8,8))
        
        ax.plot(num_estimators, scores, ".b")
        ax.set_xlabel("Number of random forest estimators")
        ax.set_ylabel("Score (R^2) / 1")
        fig.suptitle("Score of Random Forest for Different Number of Estimators")
        fig.savefig("random_forest_score.png")
        plt.close()
        

if __name__ == '__main__':
    main()
