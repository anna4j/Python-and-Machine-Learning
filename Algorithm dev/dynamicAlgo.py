import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#
# def split_excel(data, column_name, column_index):
#     pass
path = r"C:\Users\anush\Downloads\MOCK_DATA (1).xlsx"
df = pd.read_excel(path)


def split_excel(data, column_name, column_index=0, scatter=False):
    raw = data
    # dropIndex = (np.where(np.abs(stats.zscore(raw)) > 3)[0]).tolist()
    # df = raw.drop(dropIndex)
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
    #
    # Rt = df[df[column] > x]
    # L = df[df[column] <= x]
    # Ut = df[df[index] > y]
    # D = df[df[index] <= y]
    #
    # U = Ut.append(raw[raw[index] == y])
    # R = Rt.append(raw[raw[column] == x])
    #
    # q1 = pd.merge(U, R, how='inner')
    # q2 = pd.merge(U, L, how='inner')
    # q3 = pd.merge(L, D, how='inner')
    # q4 = pd.merge(R, D, how='inner')
    #
    # q1Rt = q1[q1[column] > x2]
    # q1L = q1[q1[column] <= x2]
    # q1Ut = q1[q1[index] > y2]
    # q1D = q1[q1[index] <= y2]
    #
    # q1U = q1Ut.append(q1[q1[index] == y2])
    # q1R = q1Rt.append(q1[q1[column] == x2])
    #
    # q11 = pd.merge(q1U, q1R, how='inner')
    # q12 = pd.merge(q1U, q1L, how='inner')
    # q13 = pd.merge(q1L, q1D, how='inner')
    # q14 = pd.merge(q1R, q1D, how='inner')
    #
    # q2Rt = q2[q2[column] > x1]
    # q2L = q2[q2[column] <= x1]
    # q2Ut = q2[q2[index] > y2]
    # q2D = q2[q2[index] <= y2]
    #
    # q2U = q2Ut.append(q2[q2[index] == y2])
    # q2R = q2Rt.append(q2[q2[column] == x1])
    #
    # q21 = pd.merge(q2U, q2R, how='inner')
    # q22 = pd.merge(q2U, q2L, how='inner')
    # q23 = pd.merge(q2L, q2D, how='inner')
    # q24 = pd.merge(q2R, q2D, how='inner')
    #
    # q3Rt = q3[q3[column] > x1]
    # q3L = q3[q3[column] <= x1]
    # q3Ut = q3[q3[index] > y1]
    # q3D = q3[q3[index] <= y1]
    #
    # q3U = q3Ut.append(q3[q3[index] == y1])
    # q3R = q3Rt.append(q3[q3[column] == x1])
    #
    # q31 = pd.merge(q3U, q3R, how='inner')
    # q32 = pd.merge(q3U, q3L, how='inner')
    # q33 = pd.merge(q3L, q3D, how='inner')
    # q34 = pd.merge(q3R, q3D, how='inner')
    #
    # q4Rt = q4[q4[column] > x2]
    # q4L = q4[q4[column] <= x2]
    # q4Ut = q4[q4[index] > y1]
    # q4D = q4[q4[index] <= y1]
    #
    # q4U = q4Ut.append(q4[q4[index] == y1])
    # q4R = q4Rt.append(q4[q4[column] == x2])
    #
    # q41 = pd.merge(q4U, q4R, how='inner')
    # q42 = pd.merge(q4U, q4L, how='inner')
    # q43 = pd.merge(q4L, q4D, how='inner')
    # q44 = pd.merge(q4R, q4D, how='inner')

    temp_dict = {'x':x,'x0':x0,'x1':x1,'x2':x2 ,'x3':x3,'y':y,'y0':y0,'y1':y1,'y2':y2,'y3':y3}
    return temp_dict



def cluster(data, index, temp_dict,column_name):
    temp_dict = temp_dict
    column = column_name
    raw_copy = index
    raw_copy[column] = ''
    # data = df
    # val = 0
    # ind = 0
    for i in range(len(data.values)):
        for j in range(1,(len(data.]+columns))):
            val = data.values[i][j]
            ind = data.values[i][0]
            # raw_copy = pd.DataFrame(index)

            # print(val,' ',ind)
            if ind >= temp_dict['y2'] and ind <= temp_dict['y3']:
                if val >= temp_dict['x2'] and val <=temp_dict['x3']:
                    raw_copy.loc[raw_copy['Index']==ind,column] = 'q11'
                if val >= temp_dict['x'] and val <=temp_dict['x2']:
                    # print('q12')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q12'
                if val >= temp_dict['x1'] and val <=temp_dict['x']:
                    # print('q21')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q21'
                if val >= temp_dict['x0'] and val <=temp_dict['x1']:
                    # print('q22')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q22'
            if ind >= temp_dict['y'] and ind <= temp_dict['y2']:
                if val >= temp_dict['x2'] and val <=temp_dict['x3']:
                    # print('q14')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q14'
                if val >= temp_dict['x'] and val <=temp_dict['x2']:
                    # print('q13')
                    raw_copy.loc[raw_copy['Index'] == ind,column] = 'q13'
                if val >= temp_dict['x1'] and val <=temp_dict['x']:
                    # print('q24')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q24'
                if val >= temp_dict['x0'] and val <=temp_dict['x1']:
                    # print('q23')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q23'
            if ind >= temp_dict['y1'] and ind <= temp_dict['y']:
                if val >= temp_dict['x2'] and val <=temp_dict['x3']:
                    # print('q41')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q41'
                if val >= temp_dict['x'] and val <=temp_dict['x2']:
                    # print('q42')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q42'
                if val >= temp_dict['x1'] and val <=temp_dict['x']:
                    # print('q31')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q31'
                if val >= temp_dict['x0'] and val <=temp_dict['x1']:
                    # print('q32')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q32'
            if ind >= temp_dict['y0'] and ind <= temp_dict['y1']:
                if val >= temp_dict['x2'] and val <=temp_dict['x3']:
                    # print('q44')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q44'
                if val >= temp_dict['x'] and val <=temp_dict['x2']:
                    # print('q43')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q43'
                if val >= temp_dict['x1'] and val <=temp_dict['x']:
                    # print('q34')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q34'
                if val >= temp_dict['x0'] and val <=temp_dict['x1']:
                    # print('q33')
                    raw_copy.loc[raw_copy['Index'] == ind, column] = 'q33'
            # raw_copy.to_excel(r'copy1.xlsx', index=False)
            # print(raw_copy)
    raw_copy.to_excel(r'copy1.xlsx', index=False)

#
# for i in range(1,(len(df.columns))):
#     col = pd.DataFrame(df[df.columns[i]])
#     data = pd.concat([index,col], axis=1)
#     cluster(data,temp_dict,data.columns[i])



index = pd.DataFrame(df['Index'])
for i in range(1,(len(df.columns))):
    col = pd.DataFrame(df[df.columns[i]])
    data = pd.concat([index, col], axis=1)
    temp_dict = split_excel(data, df.columns[i], 0)
    rw = cluster(data, index, temp_dict, df.columns[i])
