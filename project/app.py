'''
Author: Yuyao Wang, Tianao Wang
Date: 2020-5-9
'''

# flask
import math

import numpy
from flask import Flask, request, jsonify
from flask import redirect, render_template
from flask_cors import CORS
# pandas, numpy, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
import json

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    # Index.html

    return render_template("index.html")


@app.route('/test')
def test():

    return None


def OriginalData():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/wangTianAo/CSE564/master/project/nba_2017_nba_players_with_salary.csv")
    return data;

AllData = OriginalData();

@app.route('/ReturnAllData', methods=['POST','GET'])
def ReturnAllData():
    results = AllData.to_dict(orient='records')
    results = {'info': results}
    return jsonify(results)

@app.route('/resetData', methods=['POST'])
def resetData():
    global AllData
    AllData = OriginalData();
    results = {'info': 'ok'}
    return jsonify(results)

@app.route('/BurshRefreshData', methods=['POST'])
def BurshRefreshData():
    if request.method == "POST":
        global AllData
        svgid = request.values['SVGID']
        idlists = request.values['IDLIST']
        ids = idlists[1:-1].split(",")
        temp = ['']
        if ids != temp:
            data = OriginalData();
            newdata = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                                    'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                                    'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
            for index, row in data.iterrows():
                if (str(row['ID']) in ids):
                    newdata = newdata.append(row, ignore_index=True)
            AllData = newdata
        else:
            newdata = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                                'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                                'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
            AllData = newdata

        results = AllData.to_dict(orient='records')
        results = {'info': results}
        return jsonify(results)


@app.route('/ReturnRequestData', methods=['POST'])
def ReturnRequestData():
    if request.method == "POST":
        global AllData;
        data = AllData
        position = request.values['position']
        team = request.values['team']
        AgeRange = request.values['AgeRange']
        SalaryRange = request.values['SalaryRange']

        MinAge = 0;
        MaxAge = 99;

        if(AgeRange == 0):
            MinAge = 0;
            MaxAge = 20;
        if (AgeRange == 6):
            MinAge = 41;
            MaxAge = 99;
        if(AgeRange !=0 and AgeRange != 6):
            MinAge = 21 + 4* ((int)(AgeRange)-1)
            MaxAge = 24 + 4 * ((int)(AgeRange) - 1)

        MinSalary = 0;
        MaxSalary = 99;

        MinSalary = 0 + 5 * ((int)(SalaryRange))
        MaxSalary = 5 + 5 * ((int)(SalaryRange))

        result = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                              'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                              'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
        for index, row in data.iterrows():
            if((position == row['POSITION'] or position == "All") and (team in row['TEAM'] or team == "All")
            and ((row['AGE']>=MinAge and row['AGE']<=MaxAge) or AgeRange == "-1")
            and (((row['SALARY_MILLIONS'])>MinSalary and (row['SALARY_MILLIONS'])<=MaxSalary) or SalaryRange == "-1")):
                result = result.append(row, ignore_index=True)

        AllData = result
        results = result.to_dict(orient='records')
        results = {'info': results}
        return jsonify(results)

@app.route('/LoadSVG1', methods=['POST', 'GET'])
def LoadSVG1():
    data = AllData;
    dataframe = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                          'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                          'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
    dataframe = dataframe.append(data, ignore_index=True)
    svg1 = dataframe
    svg1 = svg1.loc[:,['ORPM', 'DRPM','clusterTemp','ID']]
    svg1.columns = ['x', 'y','cluster','ID']
    svg1 = svg1.to_dict(orient='records')

    results = {'svg1': svg1}
    return jsonify(results)


@app.route('/LoadSVG6', methods=['POST', 'GET'])
def LoadSVG6():
    global AllData
    data = AllData;

    dataframe = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                          'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                          'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
    dataframe = dataframe.append(data, ignore_index=True)
    svg6 = dataframe
    svg6 = svg6.loc[:,['AGE', 'TRB','AST','POINTS','ORPM','DRPM','SALARY_MILLIONS']]
    svg6.loc['avg'] = svg6.apply(lambda x: x.mean())
    svg6 = svg6.tail(1).fillna(0)
    svg6 = svg6.to_dict(orient='records')
    print(svg6)
    results = {'svg6': svg6}
    return jsonify(results)

@app.route('/LoadSVG3', methods=['POST', 'GET'])
def LoadSVG3():
    data = AllData;
    dataframe = pd.DataFrame([], columns=['ID', 'PLAYER', 'POSITION', 'AGE', 'ORB', 'DRB', 'TRB',
                                          'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'TEAM', 'GP', 'MPG', 'ORPM',
                                          'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'])
    dataframe = dataframe.append(data, ignore_index=True)

    svg3 = dataframe
    svg3 = svg3.loc[:,['RPM', 'SALARY_MILLIONS','clusterTemp','ID']]
    svg3.columns = ['x', 'y','cluster','ID']
    svg3 = svg3.to_dict(orient='records')

    results = {'svg3': svg3}
    return jsonify(results)


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f

n_hidden_1 = 128
n_input = 6
n_classes = 1
training_epochs = 300
batch_size = 3
model = Sequential()


def MLLoadData():
    data = OriginalData();
    data = data.drop(['ID', 'PLAYER', 'POSITION', 'ORB', 'DRB','STL', 'BLK', 'TOV', 'PF', 'TEAM', 'GP', 'MPG', 'RPM',  'clusterTemp'], axis=1)

    y = pd.DataFrame(data,columns=['SALARY_MILLIONS'])
    x = data.drop(['SALARY_MILLIONS'], axis=1)

    x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2)
    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.fit_transform(x_test)

    # print(x);
    global model
    model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input))
    model.add(Dense(n_classes))
    model.compile(loss='mae', optimizer='adam', metrics=['mae', r2])

    #print(x_test)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

    pred_test_y = model.predict(x_test)
    #print(y_test)
    #print(pred_test_y)

    pred_acc = r2_score(y_test, pred_test_y)
    print('pred_acc', pred_acc)

MLLoadData();
@app.route('/PredictData', methods=['POST'])
def PredictData():
    if request.method == "POST":
        age = float(request.values['age'])
        points = float(request.values['points'])
        trb = float(request.values['trb'])
        ast = float(request.values['ast'])
        orpm = float(request.values['orpm'])
        drpm = float(request.values['drpm'])

        predictData = pd.DataFrame([], columns=['AGE', 'TRB', 'AST', 'POINTS',  'ORPM','DRPM'])
        predictData = predictData.append([{'AGE': age, 'TRB': trb, 'AST': ast, 'POINTS': points, 'ORPM': orpm, 'DRPM': drpm}], ignore_index=True)
        #predictData = [[age,trb,ast,points,orpm,drpm]]
        # sc = StandardScaler()
        # predictData = sc.fit_transform(predictData)
        # print(predictData)
        global model
        pred_result = model.predict(predictData)
        pred_result = np.array(pred_result)
        # print(pred_result[0][0]);

        results = {'PredictResult': json.dumps(str(pred_result[0][0])) }
        return jsonify(results)

@app.route('/BarChartSalaryAndCount', methods=['POST', 'GET'])
def BarChartSalaryAndCount():
    data = AllData
    salaryDistribution = [0, 0, 0, 0, 0, 0];

    for index, row in data.iterrows():
        if ((row['SALARY_MILLIONS']) <= 5):
            salaryDistribution[0] += 1
        if (5 < (row['SALARY_MILLIONS']) <= 10):
            salaryDistribution[1] += 1
        if (10 < (row['SALARY_MILLIONS']) <= 15):
            salaryDistribution[2] += 1
        if (15 < (row['SALARY_MILLIONS']) <= 20):
            salaryDistribution[3] += 1
        if (20 < (row['SALARY_MILLIONS']) <= 25):
            salaryDistribution[4] += 1
        if (25 < (row['SALARY_MILLIONS'])):
            salaryDistribution[5] += 1

    dataframe = pd.DataFrame(columns=('x', 'y'))
    dataframe = dataframe.append([{'x': '<=5', 'y': salaryDistribution[0]}])
    dataframe = dataframe.append([{'x': '5-10', 'y': salaryDistribution[1]}])
    dataframe = dataframe.append([{'x': '10-15', 'y': salaryDistribution[2]}])
    dataframe = dataframe.append([{'x': '15-20', 'y': salaryDistribution[3]}])
    dataframe = dataframe.append([{'x': '20-25', 'y': salaryDistribution[4]}])
    dataframe = dataframe.append([{'x': '>25', 'y': salaryDistribution[5]}])
    result = dataframe.to_dict(orient='records')
    result = {'svg5': result}
    return jsonify(result)

@app.route('/BarChartAgeAndCount', methods=['POST', 'GET'])
def BarChartAgeAndCount():
    data = AllData
    ageDistribution = [0,0,0,0,0,0,0];

    for index,row in data.iterrows():
        if((int)(row['AGE'])<=20):
            ageDistribution[0]+= 1
        if(21 <= (int)(row['AGE']) <= 24):
            ageDistribution[1]+= 1
        if(25 <= (int)(row['AGE']) <= 28):
            ageDistribution[2]+= 1
        if(29 <= (int)(row['AGE']) <= 32):
            ageDistribution[3]+= 1
        if(33 <= (int)(row['AGE']) <= 36):
            ageDistribution[4]+= 1
        if(37 <= (int)(row['AGE']) <= 40):
            ageDistribution[5]+= 1
        if((int)(row['AGE']) >= 41):
            ageDistribution[6]+= 1

    dataframe = pd.DataFrame(columns=('x', 'y'))
    dataframe = dataframe.append([{'x': '<=20','y': ageDistribution[0]}])
    dataframe = dataframe.append([{'x': '21-24', 'y': ageDistribution[1]}])
    dataframe = dataframe.append([{'x': '25-28', 'y': ageDistribution[2]}])
    dataframe = dataframe.append([{'x': '29-32', 'y': ageDistribution[3]}])
    dataframe = dataframe.append([{'x': '33-36', 'y': ageDistribution[4]}])
    dataframe = dataframe.append([{'x': '37-40', 'y': ageDistribution[5]}])
    dataframe = dataframe.append([{'x': '>=41', 'y': ageDistribution[6]}])
    #print(dataframe)
    result = dataframe.to_dict(orient='records')
    result = {'svg4': result}
    return jsonify(result)

# n_component = 4 can represent 76% data
@app.route('/ScreePlotOriginalPCA', methods=['POST', 'GET'])
def ScreePlotOriginalPCA():
    data = OriginalData()
    data = data.drop(['ID'], axis=1)
    data = data.drop(['PLAYER'], axis=1)
    data = data.drop(['POSITION'], axis=1)
    data = data.drop(['TEAM'], axis=1)
    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=17)
    pcaData = pca.fit_transform(data)
    result = pd.DataFrame([],columns=['x','y'])
    result['x'] = list(range(1,18))
    result['y'] = list(pca.explained_variance_ratio_)
    result = result.to_dict(orient='records')
    result = {'data': result}

    return jsonify(result)

@app.route('/ScatterplotforTopTwoPCALoading', methods=['POST', 'GET'])
def ScatterplotforTopTwoPCALoading():
    data = AllData
    id_columns = data["ID"]
    clusters_columns = data["clusterTemp"]
    data = data.drop(['ID'], axis=1)
    data = data.drop(['PLAYER'], axis=1)
    data = data.drop(['POSITION'], axis=1)
    data = data.drop(['TEAM'], axis=1)

    if data.shape[0] < 4:
        result = {'svg2': "", 'xlabel': "NULL", 'ylabel': "NULL"}
        return jsonify(result)

    top2 = getTopTwoPCALoading(data, 4);
    columns = ['AGE', 'ORB', 'DRB', 'TRB',
                                          'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'GP', 'MPG', 'ORPM',
                                          'DRPM', 'RPM', 'SALARY_MILLIONS', 'clusterTemp'];
    for i in top2:
        columns.remove(i);
    data = data.drop(columns, axis=1)
    number = math.ceil(data.shape[0])
    data['cluster'] = clusters_columns
    data['ID'] = id_columns

    data.columns = ['x', 'y', 'cluster','ID']
    result = data.to_dict(orient='records')
    result = {'svg2': result, 'xlabel':top2[0],'ylabel':top2[1]}
    return jsonify(result)

def getTopTwoPCALoading(data,component):
    stddata = StandardScaler().fit_transform(data)
    pca = PCA(n_components=component)
    pcaData = pca.fit_transform(stddata)
    pcaLoading = pd.DataFrame(data=pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pcaLoading.insert(loc=0, column='Attr', value=list(data))
    pcaLoading['PCALoading'] = pcaLoading.drop(['Attr'], axis=1).apply(np.square).sum(axis=1)
    sortedPCA = pcaLoading.sort_values(by=['PCALoading'], ascending=False)[:2]
    temp = sortedPCA.values.tolist()
    result = [temp[0][0],temp[1][0]];
    return result;