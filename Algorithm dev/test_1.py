import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\anush\Downloads\MOCK_DATA (7).xlsx")
index = pd.DataFrame(df['Index'])


def split_excel(data, column_name, column_index=0):
    raw = data
    df = raw

    column = column_name
    index = raw.columns[column_index]

    x0 = df[column].min()
    x = (df[column].max() + df[column].min()) / 2
    x1 = (df[column].min() + x) / 2
    x2 = (df[column].max() + x) / 2
    x3 = df[column].max()
    y0 = raw[index].min()
    y = raw.shape[0] / 2
    y1 = y / 2
    y2 = y + y1
    y3 = raw[index].max()

    temp_dict = {'x':x,'x0':x0,'x1':x1,'x2':x2 ,'x3':x3,'y':y,'y0':y0,'y1':y1,'y2':y2,'y3':y3}
    return temp_dict


def quad(data, temp_dict, i, raw_copy):
    # raw = data
    temp_dict = temp_dict
    i = i
    name = df.columns[i]
    raw_copy[name] = ""
    column = df.columns[i]
    for j in range(len(data.values)):
        val = data.values[j][1]
        ind = data.values[j][0]
        if ind >= temp_dict['y2'] and ind <= temp_dict['y3']:
            if val >= temp_dict['x2'] and val <= temp_dict['x3']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q11'
            if val >= temp_dict['x'] and val <= temp_dict['x2']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q12'
            if val >= temp_dict['x1'] and val <= temp_dict['x']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q21'
            if val >= temp_dict['x0'] and val <= temp_dict['x1']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q22'
        if ind >= temp_dict['y'] and ind <= temp_dict['y2']:
            if val >= temp_dict['x2'] and val <= temp_dict['x3']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q14'
            if val >= temp_dict['x'] and val <= temp_dict['x2']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q13'
            if val >= temp_dict['x1'] and val <= temp_dict['x']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q24'
            if val >= temp_dict['x0'] and val <= temp_dict['x1']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q23'
        if ind >= temp_dict['y1'] and ind <= temp_dict['y']:
            if val >= temp_dict['x2'] and val <= temp_dict['x3']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q41'
            if val >= temp_dict['x'] and val <= temp_dict['x2']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q42'
            if val >= temp_dict['x1'] and val <= temp_dict['x']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q31'
            if val >= temp_dict['x0'] and val <= temp_dict['x1']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q32'
        if ind >= temp_dict['y0'] and ind <= temp_dict['y1']:
            if val >= temp_dict['x2'] and val <= temp_dict['x3']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q44'
            if val >= temp_dict['x'] and val <= temp_dict['x2']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q43'
            if val >= temp_dict['x1'] and val <= temp_dict['x']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q34'
            if val >= temp_dict['x0'] and val <= temp_dict['x1']:
                raw_copy.loc[raw_copy['Index'] == ind, column] = 'q33'

    return raw_copy

raw = pd.DataFrame()
for i in range(1,(len(df.columns))):
    col = pd.DataFrame(df[df.columns[i]])
    data = pd.concat([index, col], axis=1)
    temp_dict = split_excel(data, df.columns[i], 0)
    print(temp_dict)
    if i == 1:
        raw = data.copy()
    raw = quad(data, temp_dict, i, raw)
    # print(raw)

raw.to_excel('abc.xlsx', index=False)
#
# r = quad(pd.DataFrame(data=[1,2,3]), {}, 5, raw_copy=r)
# raw_copy = r