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


features = ['University_name', 'Region', 'Founded_year', 'Motto',
       'UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score',
       'International_students', 'Student_satisfaction', 'Student_enrollment',
       'Academic_staff', 'Control_type', 'Academic_Calender', 'Campus_setting',
       'Estimated_cost_of_living_per_year_(in_pounds)', 'Latitude',
       'Longitude', 'Website']

used_features = features = ['Founded_year',
       'UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score',
       'International_students', 'Student_satisfaction', 'Student_enrollment',
       'Academic_staff', 'Control_type', 'Academic_Calender', 'Campus_setting',
       'Estimated_cost_of_living_per_year_(in_pounds)', 'Latitude',
       'Longitude']
def main():
    # Specify the path to the CSV file
    csv_file_path = 'Universities.csv'

    global used_features
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    df = preprocessing(df)
    df_normalized = normalize(df)
    
    target1 = 'UG_average_fees_(in_pounds)'
    plotting_continuos_features(df, target1)

    # Split the data into independent variables (X) and the dependent variable (y)
    X = df_normalized[used_features]  # Replace feature1, feature2, feature3 with your actual column names
    y = df_normalized[target1]  # Replace target_variable with your actual column name

    print(pd.isnull(X).sum())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42)
    
    data = pd.DataFrame({
    'Latitude': [40.7128, 34.0522, 37.7749, 29.7604],
    'Longitude': [-74.0060, -118.2437, -122.4194, -95.3698],
    'value': [10, 20, 15, 25]
    })
    
    #plotHeatMap(df["Latitude"], df["Longitude"], y)


    #linearRegression(X_train, X_test, y_train, y_test)
    supportVectorRegression(X, y)
    #random_forest_regression(X_train, X_test, y_train, y_test)
    multiLayerPerceptron(X, y)

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

def plot_relationship(data, var1, var2, ax=None, xlabel=None, ylabel=None):
    # Create scatter plot of two variables using Matplotlib
    if ax is None:
        plt.scatter(data[var1], data[var2])
        plt.title('{} - {}'.format(var1, var2))
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.savefig('plots/scatter-{}-{}-1.png'.format(var1, var2))
        plt.close()
    else:
        if xlabel=="Estimated_cost_of_living_per_year_(in_pounds)":
            xlabel="Yearly cost of living / £"
        ax.scatter(data[var1], data[var2], s=1)
        ax.set_xlabel(xlabel)# should be a feature
        ax.set_ylabel(ylabel)# should be the target
     

    #plot_contourplot(data, var1, var2)


def plotting_continuos_features(data, target):
    discrete_features = ['Control_type', 'Academic_Calender', 'Campus_setting', 'Region']
    dont_plot = ['Unnamed: 0', 'PG_average_fees_(in_pounds)', target]
    fig, axs = plt.subplots(3, 4)
    fig.set_size_inches((11.7*1.3,8.3*1.3))
    axs=axs.flatten()
    fig.suptitle("Scatter Plot of Continuous Features with Target")
    
    axs_idx = 0
    for idx,x in enumerate(data.columns):
        if x not in dont_plot and x not in discrete_features:
            ylabel = None
            if axs_idx in [0,4,8]:
                ylabel="tuition fee"
                axs[axs_idx].set_yticks(np.linspace(0,np.max(data[target]),5))
            else:
                axs[axs_idx].set_yticks([])
            plot_relationship(data, x, target, ax=axs[axs_idx], xlabel=x, ylabel=ylabel)
    
            axs_idx += 1
    fig.savefig("scatter_plot_feature_vs_target.png")
    plt.close()

def plotHeatMap(lat, long, target):
    # Create a pivot table to reshape the data for the heatmap
    heatmap_data = plt.scatter(x=long, y=lat, c=target)

    # Plot the heatmap using Matplotlib
    plt.colorbar(label='Value')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot')
    plt.show()

def range_mean(stringList, axis):
    if stringList[0] != "NaN":
        if stringList[0] != "over":
            return int((int(stringList[0]) + int(stringList[1])) / 2)
        else:
            return int(stringList[1])
    else:
        return 0

def founded_year_filter(year, axis):
    if year == 9999:
        return 0
    if year == np.nan:
        return 0
    return 2023 - year

def preprocessing(data):
    data = data.drop(['University_name', 'Website', 'Motto'], axis=1)
    data = data.drop_duplicates()
    # Founded_year is all over the plays
    data['Founded_year'] = data['Founded_year'].iloc[:].apply(founded_year_filter, axis=1)
    data["Founded_year"].iloc[:][pd.isnull(data["Founded_year"])] = 0
    data["Academic_Calender"].iloc[:][pd.isnull(data["Academic_Calender"])] = "other"
    data["Campus_setting"].iloc[:][pd.isnull(data["Campus_setting"])] = "other"
    data["CWUR_score"].iloc[:][pd.isnull(data["CWUR_score"])] = data["CWUR_score"].mean()
    #print(data['Founded_year'])
    # convert 10.00% and over-1000 into int an float
    prozent_col = ['International_students', 'Student_satisfaction', ]
    ranges_col = ['Student_enrollment', 'Academic_staff']
    for pro_col in prozent_col:
        data[pro_col] = data[pro_col].str.replace("%","")
        data[pro_col] = pd.to_numeric(data[pro_col])
    for rang_col in ranges_col:
        data[rang_col] = data[rang_col].str.replace(",","")
        data[rang_col] = data[rang_col].str.split("-")
        data[rang_col] = data[rang_col].apply(range_mean, axis=1)
    # categorical features to numerical
    label_encoder = LabelEncoder()
    mappings = {}
    for x in data.columns:
        col = data[x].loc[:]
        if is_object_dtype(col):
            #print(col)
            # print(x, " is object-converting")
            col[pd.isnull(col)] = 'NaN'
            #print(col)
            data[x] = col
            data[x] = label_encoder.fit_transform(data[x])
            #print(data[x])
            le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            #print(le_name_mapping)
            if 'NaN' in le_name_mapping:
                data[x] = data[x].replace([le_name_mapping['NaN']], np.nan)

            mappings[x] = le_name_mapping
            #print(le_name_mapping)
    #print(data)
    setMissing(data)
    return data

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

def setMissing(df):
    df["Founded_year"] = df["Founded_year"].replace([9999], np.nan)

def linearRegression( X_train, X_test, y_train, y_test):
    print("Linear Regression: ")


def multiLayerPerceptron(X, y):
    params = {
        "solver": ["lbfgs", "adam"],
        "learning_rate_init": [0.005,0.005, 0.01, 0.1, 0.3],
        "hidden_layer_sizes": [(3,3), (12,),(10,)],
        "alpha": [1e-4, 1e-5, 1e-6,5e-6, 1e-7]
    }
    model = MLPRegressor( random_state=1, max_iter=1000)
    cv = GridSearchCV(model, params, cv=5, n_jobs=-1)
    cv.fit(X, y)

    print(f"best score: {cv.best_score_} with params {cv.best_params_}")


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
 
def supportVectorRegression( X, y):
    print("Support Vector Regression:")
    print(X.shape)
    rfr = SVR()
    params = {
        "kernel": ["linear", "poly"],
        "C": [0.005, 0.01, 0.1, 1.0, 3, 10.],
        "epsilon": [0.01, 0.05, 0.1, 1],
        "degree": [2, 3, 4]
    }
    cv = GridSearchCV(rfr, params, n_jobs=-1, cv=5)
    cv.fit(X=X, y=y)

    print(f"best score: {cv.best_score_} with params {cv.best_params_}")
   
def random_forest_regression(X_train, X_test, y_train, y_test):
    PLOT_SCORE_OVER_N_EST = True
    
    print(f"Using {X_train.shape[1]} features for random forest regression")
    scores = list()
    
    num_estimators = np.arange(10, 200, 5)    
    
    for num_estimator in num_estimators:
        rf_regressor = RandomForestRegressor(n_estimators=num_estimator)
        rf_regressor.fit(X_train, y_train)
        score = rf_regressor.score(X_test, y_test)
        print(f"Num Estimators = {num_estimator}, Score = {score}")
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
