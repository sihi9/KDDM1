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

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, mean_squared_error
import geopandas as gpd

features = ['University_name', 'Region', 'Founded_year', 'Motto',
       'UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score',
       'International_students', 'Student_satisfaction', 'Student_enrollment',
       'Academic_staff', 'Control_type', 'Academic_Calender', 'Campus_setting',
       'Estimated_cost_of_living_per_year_(in_pounds)', 'Latitude',
       'Longitude', 'Website']

used_features = features = [
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
    target2 = 'PG_average_fees_(in_pounds)'
    #plot_total_heatmap(df)
    plotting_features(df, target1)
    #used_features = ['UK_rank', 'World_rank', 'CWUR_score', 'Minimum_IELTS_score']
    # Split the data into independent variables (X) and the dependent variable (y)
    X = df_normalized[used_features]  # Replace feature1, feature2, feature3 with your actual column names
    y = df_normalized[target1]  # Replace target_variable with your actual column name

    #plot_pivot(df[["Region", "Academic_Calender"]], index="Region", column= "Academic_Calender")
    print(pd.isnull(X).sum())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42)
    
    data = pd.DataFrame({
    'Latitude': [40.7128, 34.0522, 37.7749, 29.7604],
    'Longitude': [-74.0060, -118.2437, -122.4194, -95.3698],
    'value': [10, 20, 15, 25]
    })

    #plotHeatMap(df["Latitude"], df["Longitude"], y)
    #findOptimalRegressionModel(X_train, X_test, y_train, y_test)
    #random_forest_regression(X_train, X_test, y_train, y_test)
    multiLayerPerceptron(X_train, y_train,X_test,y_test)
    supportVectorRegression(X_train, y_train,X_test,y_test)
    #scorer = make_scorer(mean_squared_error, greater_is_better=False)
    elasticnet_regression(X_train, X_test, y_train, y_test)


def findOptimalRegressionModel(X_train, X_test, y_train, y_test):
    scorer = make_scorer(mean_squared_error, greater_is_better=False) # not used atm (instead r2 is used)

    linearRegression(X_train, X_test, y_train, y_test)
    differentRegulationLinearRegressionModels(X_train, X_test, y_train, y_test, scorer)

    X_train_pca, X_test_pca = principalComponentAnalysis(X_train, X_test)
    print("------------------with pca------------------")
    differentRegulationLinearRegressionModels(X_train_pca, X_test_pca, y_train, y_test, scorer)

    supportVectorRegression(X_train, y_train, X_test, y_test, scorer)
    print("------------------with pca------------------")
    supportVectorRegression(X_train_pca, y_train, X_test_pca, y_test, scorer)


def differentRegulationLinearRegressionModels(X_train, X_test, y_train, y_test, scorer):
    ridge_regression(X_train, X_test, y_train, y_test, scorer)
    lasso_regression(X_train, X_test, y_train, y_test, scorer)
    elasticnet_regression(X_train, X_test, y_train, y_test, scorer)

def interpolate(df, target_column, predictor_columns):
    # Split the DataFrame into two subsets: one with non-missing values and one with missing values
    non_missing_subset = df.dropna(subset=[target_column])
    missing_subset = df[df[target_column].isna()]

    # Perform linear regression on the non-missing subset
    regressor = LinearRegression()
    regressor.fit(non_missing_subset[predictor_columns], non_missing_subset[target_column])

    # Use the regression model to predict the missing values
    predicted_values = regressor.predict(missing_subset[predictor_columns])

    # Replace the missing values in the original DataFrame with the predicted values
    df.loc[df[target_column].isna(), target_column] = predicted_values

    print(df)


def principalComponentAnalysis(X_train, X_test):
    """
    # find optimal nr of components
    X_pca = PCA().fit(X_train)
    # Explained variance ratio
    explained_var_ratio = X_pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_var_ratio)

    plt.plot(np.cumsum(X_pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    plt.close()
    """

    # Perform PCA
    n_components = 6
    pca = PCA(n_components=n_components)  # Set the number of components you want to retain
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)



    return X_train_pca, X_test_pca


def plot_total_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(16, 16))
    sns.heatmap(corr, cmap='rainbow', annot=True)

    plt.show()
    # most important

def plot_pivot(df, index, column):
    data = pd.pivot_table(df, index = index, columns=column, aggfunc=len, fill_value=0)
    f, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(data, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.savefig(f'plots/heatmap-{index}-{column}.png')
    #plt.show()
    plt.close()

def get_ranks_correlation(df):
    df["UK_inv"] = 1/df["UK_rank"]
    df["World_inv"] = 1/df["World_rank"]
    corr = df[['UK_rank', 'World_rank',"UK_inv", "World_inv", 'CWUR_score', 'UG_average_fees_(in_pounds)']].corr()
    #print(corr)
    corr.to_csv("correlations.csv", float_format='%.3f')

def plot_locations(df):
    # Load the world shapefile using geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # then restrict this to the United Kingdom
    ax = world[(world.name == "United Kingdom")].plot(
    color='white', edgecolor='black')

    df = pd.read_csv("Universities.csv")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

    # then plot the geodataframe on this
    gdf.plot(ax=ax, alpha=0.5, markersize=10)

    # Customize the plot
    plt.title('')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

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


def plotting_features(data, target):
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
    fig.savefig("plots/scatter_plot_feature_vs_target.png")
    plt.close()
    
    #now discrete features
    sns.boxplot(x="Academic_Calender",y=target, data=data)
    plt.savefig("plots/academic_calender_vs_target.png")
    plt.close()
    sns.boxplot(x="Campus_setting",y=target, data=data)
    plt.savefig("plots/campus_setting_vs_target.png")
    plt.close()
    sns.boxplot(x="Region",y=target, data=data)
    plt.savefig("plots/region_vs_target.png")
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
        return np.nan
    return year

def preprocessing(data):
    data = data.drop(['University_name', 'Website', 'Motto','Founded_year'], axis=1)
    data = data.drop_duplicates()
    # Founded_year is all over the plays
    #data['Founded_year'] = data['Founded_year'].iloc[:].apply(founded_year_filter, axis=1)
    #data["Founded_year"].iloc[:][pd.isnull(data["Founded_year"])] = 0
    data["Academic_Calender"].iloc[:][pd.isnull(data["Academic_Calender"])] = "nothing"
    data["Campus_setting"].iloc[:][pd.isnull(data["Campus_setting"])] = "other"
    data["CWUR_score"] = data["CWUR_score"].interpolate(method="pad",axsi=1)

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
            print(le_name_mapping)
    #print(data)
    #setMissing(data)
    return data

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

def setMissing(df):
    df["Founded_year"] = df["Founded_year"].replace([9999], np.nan)



def multiLayerPerceptron(X_train, y_train, X_test, y_test):
    params = {
        "solver": ["adam"],
        "learning_rate_init": [0.005,0.005, 0.01, 0.1, 0.3],
        #"hidden_layer_sizes": [(3,3), (12,),(10,)],
        "hidden_layer_sizes": [(3,3), (3,3,3), (6,6), (10,10)],
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    }
    model = MLPRegressor(activation="relu", random_state=1, max_iter=1000,
                         batch_size=32, shuffle=True)
    cv = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
    cv.fit(X_train, y_train)

    print(f"best score: {cv.best_score_} with params {cv.best_params_}")
    print('Test score for MLP: ', cv.score(X_test, y_test))


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



def ridge_regression(X_train, X_test, y_train, y_test, scorer="neg_mean_squared_error"):
    # list of alpha to tune
    params = {
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
           7.0, 8.0, 9.0,
           10.0, 20, 50, 100, 500, 1000]}

    ridge = Ridge()
    cv = GridSearchCV(estimator=ridge,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5,
                      return_train_score=True,
                      verbose=1)
    cv.fit(X_train, y_train)

    print(f"best score: {cv.best_score_} with params {cv.best_params_} for ridge")
    print('Test score for ridge: ', cv.score(X_test, y_test))


def lasso_regression(X_train, X_test, y_train, y_test, scorer="neg_mean_squared_error"):
    # list of alpha to tune
    params = {
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
           7.0, 8.0, 9.0,
           10.0, 20, 50, 100, 500, 1000]}
    lasso = Lasso()
    cv = GridSearchCV(estimator=lasso,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5,
                      return_train_score=True,
                      verbose=1)

    cv.fit(X_train, y_train)

    print(f"best score: {cv.best_score_} with params {cv.best_params_} for lasso")
    print('Test score for lasso: ', cv.score(X_test, y_test))


def elasticnet_regression(X_train, X_test, y_train, y_test, scorer="r2"):
    # list of alpha to tune
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0,
                  10.0, 20, 50, 100, 500, 1000]}

    elasticnet = ElasticNet()
    cv = GridSearchCV(estimator=elasticnet,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5,
                      return_train_score=True,
                      verbose=1)
    cv.fit(X_train, y_train)

    print(f"best score: {cv.best_score_} with params {cv.best_params_} for elastic")
    print('Test score for elasticnet: ', cv.score(X_test, y_test))


def supportVectorRegression( X, y, X_test, y_test, scorer="neg_mean_squared_error"):
    print("Support Vector Regression:")
    print(X.shape)
    rfr = SVR()
    params = {
        "kernel": ["linear", "poly","sigmoid"],
        "C": [0.005, 0.01, 0.1, 1.0, 3, 10.],
        "epsilon": [0.01, 0.05, 0.1, 1],
        "degree": [2, 3, 4, 5]
    }
    cv = GridSearchCV(rfr, params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    cv.fit(X=X, y=y)

    print(f"best score: {cv.best_score_} with params {cv.best_params_} for svr")
    print('Test score for support vector regression: ', cv.score(X_test, y_test))

   
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
