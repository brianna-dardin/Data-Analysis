""""
All code was written by Brianna Dardin

The main functions (big_query, viz_data, pop_data, mining_data, decision_tree, cluster)
are meant to be run independently of each other as they were done separately.
The code was consolidated into one file for submission.
""""

# all the required imports
import os
from google.cloud import bigquery
import pandas as pd
import pickle
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# I believe this function requires the use of a virtual environment in order to run
# the bigquery library specifically didn't work otherwise
def big_query():
    # set the environment variable for the Google credentials to be the API key in JSON format
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),'google-key.json')
    
    # connect to the BigQuery database
    client = bigquery.Client()
    dataset = client.dataset('nhtsa_traffic_fatalities', project='bigquery-public-data')

    # the database has many tables but for simplicity we only looked at the main accident tables
    years = ['2015','2016']
    df = pd.DataFrame()
    for y in years:
        table_name = 'accident_' + y
        table_ref = dataset.table(table_name)
        table = client.get_table(table_ref)
        rows = client.list_rows(table)
        
        table_df = rows.to_dataframe()
        df = pd.concat([df,table_df], ignore_index=True)
            
    df.to_csv("accidents.csv", index=False)

# this function produced the base dataset that was used for visualizations    
def viz_data():
    # read data into dataframe
    df = pd.read_csv('accidents.csv')
    
    # check each column for missing values and calculate percentage missing
    null_count = pd.isnull(df).sum()
    null_df = pd.DataFrame({'column':null_count.index, 'total_nulls':null_count.values})
    
    for i, row in null_df.iterrows():
        if row['total_nulls'] > 0:
            print(row)
    
    # only one column has missing values so we can drop that column
    df.drop('trafficway_identifier_2', axis=1, inplace=True)
    
    # check number of unique values present in each column
    for col in df.columns:
        uniques = df[col].unique()
        
        # if a column has only 1 unique value or if each row has a unique value
        # drop the column as it won't contribute anything to the analysis
        if len(uniques) == 1 or len(uniques) == df.shape[0]:
            df.drop(col, axis=1, inplace=True)
        # otherwise write columns, total number of unique values and actual values
        # to a text file for quick manual analysis
        else:
            with open('uniques.txt', 'a') as txt:
                txt.write(col+': '+str(len(uniques))+'\n'+str(uniques)+'\n\n')
            
    # after manual analysis, determined the following columns to have either too many
    # unique values to be useful OR whose values were not explained in the description
    # and could not be intuited OR were redundant. Most were redundant, with one column
    # having a numeric identifier and another column with the description
    cols_to_drop = ['state_number','consecutive_number','trafficway_identifier','land_use',
                    'functional_system','ownership','route_signing','milepoint',
                    'special_jurisdiction','first_harmful_event','manner_of_collision',
                    'relation_to_junction_specific_location','relation_to_trafficway',
                    'light_condition','atmospheric_conditions_1','atmospheric_conditions_2',
                    'atmospheric_conditions','rail_grade_crossing_identifier',
                    'related_factors_crash_level_1','related_factors_crash_level_2',
                    'related_factors_crash_level_3','city','county',
                    'number_of_vehicle_forms_submitted_all','number_of_parked_working_vehicles',
                    'number_of_motor_vehicles_in_transport_mvit',
                    'number_of_forms_submitted_for_persons_not_in_motor_vehicles',
                    'number_of_persons_not_in_motor_vehicles_in_transport_mvit',
                    'number_of_persons_in_motor_vehicles_in_transport_mvit',
                    'number_of_forms_submitted_for_persons_in_motor_vehicles']
    
    # drop these additional columns
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    #70 columns reduced to 39

    # make boolean field a little more readable for visual analysis
    df['national_highway_system'] = df['national_highway_system'].map({1 : 'Yes', 0 : 'No', 9 : 'Unknown'})  
    
    # create a pure date field for time series analysis
    df['date_of_crash'] = df.apply(date_of_crash,axis=1)
    
    # finally export dataframe to CSV
    df.to_csv('viz_data.csv', index=False)

# this function converts the timestamp into a simple date    
def date_of_crash(x):
    try:
        crash = x['timestamp_of_crash'].split("+")[0]
        date = datetime.strptime(crash, "%Y-%m-%d %H:%M:%S").date()
        return date
    except:
        return np.nan

# this pulls the population estimates for each state for both years
# then compares them to the amount of crashes in each state
def pop_data():
    df = pd.read_csv('viz_data.csv')
    
    # I couldn't figure out a way to pivot the data in the way I wanted without adding
    # this column that is 1 for every row
    df['count'] = df.apply(lambda x: fake(x), axis=1)
    
    # this pivots the data by state and year
    pivot = pd.pivot_table(df, values=['count'], index=['state_name'], columns=['year_of_crash'], aggfunc=np.sum)
    pivot.columns = pivot.columns.to_series().str.join('_')
    pivot.reset_index(inplace=True)
    pivot.columns = ['state_name','2015','2016']
    
    # data from https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html
    pop_df = pd.read_csv("nst-est2018-alldata.csv")
    
    # this merges the population data with the pivot data
    new_df = pd.merge(pivot,pop_df[['NAME', 'POPESTIMATE2015', 'POPESTIMATE2016']],left_on='state_name', right_on='NAME', how='left')
    new_df.drop('NAME',axis=1,inplace=True)
    
    # this produces the per capita percentages
    new_df['2015_pct'] = new_df.apply(lambda x: x['2015'] / x['POPESTIMATE2015'],axis=1)
    new_df['2016_pct'] = new_df.apply(lambda x: x['2016'] / x['POPESTIMATE2016'],axis=1)
    
    new_df.to_csv('population.csv',index=False)

def fake(x):
    return 1

# this function builds off the visualization data and prepares it for the data mining tasks    
def mining_data():
    df = pd.read_csv('viz_data.csv')
    
    # these binary fields are created since we were more interested in seeing if there was
    # any effect from having more than one fatality or any drunk driving involvement
    # the exact number of fatalities/drunk drivers did not seem useful
    df['fatality_binary'] = df.apply(lambda x: binary(x['number_of_fatalities'],1),axis=1)
    df['drunk_binary'] = df.apply(lambda x: binary(x['number_of_drunk_drivers'],0),axis=1)
    
    # each of the following categorical variables were created in order to reduce the
    # number of values each had and therefore the number of dummies created
    df['weather'] = df.apply(lambda x: weather(x['atmospheric_conditions_name']),axis=1)
    df['light'] = df.apply(lambda x: light(x['light_condition_name']),axis=1)
    df['collision'] = df.apply(lambda x: collision(x['manner_of_collision_name']),axis=1)
    df['land_use'] = df.apply(lambda x: land_use(x['land_use_name']),axis=1)
    df['route'] = df.apply(lambda x: route(x['route_signing_name']),axis=1)
    
    # instead of creating dummies for each state we created variables for their
    # corresponding US Census defined regions
    state_df = pd.read_csv('https://raw.githubusercontent.com/cphalpert/census-regions/master/us%20census%20bureau%20regions%20and%20divisions.csv')
    new_df = pd.merge(df,state_df[['State','Division']],left_on='state_name', right_on='State', how='left')
    
    # this time it was easier to create a list of fields we wanted to keep
    # as we were not interested in seeing the effects of the other fields
    cols_to_keep = ['day_of_crash', 'month_of_crash', 'year_of_crash','day_of_week',
                    'route', 'land_use', 'Division','drunk_binary','light','weather',
                    'fatality_binary','collision']
    
    # this creates the new dataframe and dummy variables
    model_df = new_df[cols_to_keep].copy()
    model_df = pd.get_dummies(model_df, drop_first=True)
    
    # this creates a correlation matrix which was analyzed in excel
    # the columns dropped had a very high correlation with another variable that
    # represented a value from the same original categorical variable
    # therefore keeping both would have been redundant
    model_df.corr().to_excel('correlation_matrix.xlsx')
    model_df = model_df.drop(['land_use_Rural','light_Dark – Not Lighted'], axis=1)
    
    # this preserves the dataframe as a Python object
    with open("model-df.txt", "wb") as fp:
        pickle.dump(model_df, fp)
    
def binary(x,num):
    if x > num:
        return 1
    else:
        return 0
    
def weather(x):
    types = ["Clear","Cloudy","Rain"]
    for t in types:
        if t in x:
            return t
    return "Other"

def light(x):
    if "Daylight" in x or "Lighted" in x:
        return x
    else:
        return "Other"
    
def collision(x):
    if "Collision" in x:
        return "Not Collision"
    elif "Angle" in x or "Front" in x:
        return x
    else:
        return "Other"
    
def land_use(x):
    if "Rural" in x or "Urban" in x:
        return x
    else:
        return "Other"
    
def route(x):
    if "Local" in x:
        return "Local"
    elif "Other" in x or "Unknown" in x:
        return "Other"
    else:
        return x

# this trains the decision tree and generates the confusion matrix and tree structure graphics    
def decision_tree():
    with open("model-df.txt", "rb") as fp:
        model_df = pickle.load(fp)
    
    # the target variable is the previously created fatality binary    
    X = model_df.drop('fatality_binary', axis=1)
    y = model_df['fatality_binary']
    
    # this creates the train and test sets
    # the y variables were stratified since the class is so unbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    
    # again the class weight is set to balanced due to the unbalanced class
    # max depth was set to 2 because increasing it up to 10 led to no improvements
    tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced").fit(X_train, y_train)
    tree_predictions = tree.predict(X_test)
    
    # print all the performance metrics
    print("accuracy",accuracy_score(y_test,tree_predictions))
    print("precision",precision_score(y_test,tree_predictions))
    print("recall",recall_score(y_test,tree_predictions))
    print("f1 score",f1_score(y_test,tree_predictions))
    
    classes = ['Only One','More Than One']
    
    # this creates the dot file that can be used by graphviz to generate the tree
    # I ultimately used the command line to create the image using the dot file
    dot_data = export_graphviz(tree,out_file=None,feature_names=X.columns,
                               class_names=classes,filled=True, 
                               rounded=True,special_characters=True,label='none',
                               impurity=False)  

# this runs the cluster analysis
def cluster():
    with open("model-df.txt", "rb") as fp:
        model_df = pickle.load(fp)
    
    # I was concerned that the time fields may impact the clusters too much
    # especially since we found there wasn't too strong of a seasonal component
    cluster_df = model_df.drop(['day_of_crash','month_of_crash','year_of_crash','day_of_week'],axis=1)
    
    # we kept it simple and looked at only 3 groups
    kmeans = KMeans(n_clusters=3, random_state=0).fit(cluster_df)
    centers = pd.DataFrame(kmeans.cluster_centers_,columns=cluster_df.columns)
    
    # the average of all the columns in the original dataframe is added as an additional row
    # this way they can be compared to the averages in the centers
    centers = centers.append(cluster_df.mean(), ignore_index=True)
    centers.to_excel('centers.xlsx',index=False)

# in case you want to just simply run the file instead of running things separately
if __name__ == "__main__":
    big_query()
    viz_data() 
    pop_data()
    mining_data()
    decision_tree()
    cluster()