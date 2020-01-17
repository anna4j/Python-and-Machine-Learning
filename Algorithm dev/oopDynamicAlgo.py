import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def split_excel(path, column_name, column_index=0, scatter=False):
    raw = pd.read_excel(path)
    dropIndex = (np.where(np.abs(stats.zscore(raw)) > 3)[0]).tolist()
    df = raw.drop(dropIndex)

    column = column_name
    index = raw.columns[column_index]
    i = 1
    for i in range(len(raw.columns)):



        x0 = df[df.columns[i]].min()
        x = (df[df.columns[i]].max() + df[df.columns[i]].min()) / 2
        x1 = (df[df.columns[i]].min() + x) / 2
        x2 = (df[df.columns[i]].max() + x) / 2
        x3 = df[df.columns[i]].max()
        y0 = raw[index].min()
        y = raw.shape[0] / 2
        y1 = y / 2
        y2 = y + y1
        y3 = raw[index].max()

        Rt = df[df[df.columns[i]] > x]
        L = df[df[df.columns[i]] <= x]
        Ut = df[df[index] > y]
        D = df[df[index] <= y]

        U = Ut.append(raw[raw[index] == y])
        R = Rt.append(raw[raw[raw.columns[i]] == x])

        q1 = pd.merge(U, R, how='inner')
        q2 = pd.merge(U, L, how='inner')
        q3 = pd.merge(L, D, how='inner')
        q4 = pd.merge(R, D, how='inner')

        q1Rt = q1[q1[q1.columns[i]] > x2]
        q1L = q1[q1[q1.columns[i]] <= x2]
        q1Ut = q1[q1[index] > y2]
        q1D = q1[q1[index] <= y2]

        q1U = q1Ut.append(q1[q1[index] == y2])
        q1R = q1Rt.append(q1[q1[q1.columns[i]] == x2])

        q11 = pd.merge(q1U, q1R, how='inner')
        q12 = pd.merge(q1U, q1L, how='inner')
        q13 = pd.merge(q1L, q1D, how='inner')
        q14 = pd.merge(q1R, q1D, how='inner')

        q2Rt = q2[q2[q2.columns[i]] > x1]
        q2L = q2[q2[q2.columns[i]] <= x1]
        q2Ut = q2[q2[q2.columns[i]] > y2]
        q2D = q2[q2[q2.columns[i]] <= y2]

        q2U = q2Ut.append(q2[q2[index] == y2])
        q2R = q2Rt.append(q2[q2[q2.columns[i]] == x1])

        q21 = pd.merge(q2U, q2R, how='inner')
        q22 = pd.merge(q2U, q2L, how='inner')
        q23 = pd.merge(q2L, q2D, how='inner')
        q24 = pd.merge(q2R, q2D, how='inner')

        q3Rt = q3[q3[q3.columns[i]] > x1]
        q3L = q3[q3[q3.columns[i]] <= x1]
        q3Ut = q3[q3[index] > y1]
        q3D = q3[q3[index] <= y1]

        q3U = q3Ut.append(q3[q3[index] == y1])
        q3R = q3Rt.append(q3[q3[q3.columns[i]] == x1])

        q31 = pd.merge(q3U, q3R, how='inner')
        q32 = pd.merge(q3U, q3L, how='inner')
        q33 = pd.merge(q3L, q3D, how='inner')
        q34 = pd.merge(q3R, q3D, how='inner')

        q4Rt = q4[q4[q4.columns[i]] > x2]
        q4L = q4[q4[q4.columns[i]] <= x2]
        q4Ut = q4[q4[index] > y1]
        q4D = q4[q4[index] <= y1]

        q4U = q4Ut.append(q4[q4[index] == y1])
        q4R = q4Rt.append(q4[q4[q4.columns[i]] == x2])

        q41 = pd.merge(q4U, q4R, how='inner')
        q42 = pd.merge(q4U, q4L, how='inner')
        q43 = pd.merge(q4L, q4D, how='inner')
        q44 = pd.merge(q4R, q4D, how='inner')

    # q11.to_csv('q11.csv', index=False)
    # q12.to_csv('q12.csv', index=False)
    # q13.to_csv('q13.csv', index=False)
    # q14.to_csv('q14.csv', index=False)
    # q21.to_csv('q21.csv', index=False)
    # q22.to_csv('q22.csv', index=False)
    # q23.to_csv('q23.csv', index=False)
    # q24.to_csv('q24.csv', index=False)
    # q31.to_csv('q31.csv', index=False)
    # q32.to_csv('q32.csv', index=False)
    # q33.to_csv('q33.csv', index=False)
    # q34.to_csv('q34.csv', index=False)
    # q41.to_csv('q41.csv', index=False)
    # q42.to_csv('q42.csv', index=False)
    # q43.to_csv('q43.csv', index=False)
    # q44.to_csv('q44.csv', index=False)

    temp_dict = {'x':x,'x0':x0,'x1':x1,'x2':x2 ,'x3':x3,'y':y,'y0':y0,'y1':y1,'y2':y2,'y3':y3}
    if scatter:

        plt.scatter(df[df.columns[i]], df[index])
        plt.plot(x * np.ones(shape=(df.shape[0], 2)), df[index])
        plt.plot(df[df.columns[i]], y * np.ones(shape=(df.shape[0], 2)))
        plt.plot(df[df.columns[i]], y1 * np.ones(shape=(df.shape[0], 2)))
        plt.plot(df[df.columns[i]], y2 * np.ones(shape=(df.shape[0], 2)))
        plt.plot(x1 * np.ones(shape=(df.shape[0], 2)), df[index])
        plt.plot(x2 * np.ones(shape=(df.shape[0], 2)), df[index])
        plt.show()
    return temp_dict

M = pd.read_excel(r"C:\Users\anush\Downloads\MOCK_DATA (1).xlsx")

for j in range(1,len(M.columns)):

    Q = split_excel(r"C:\Users\anush\Downloads\MOCK_DATA (1).xlsx", M.columns[j], 0, scatter=True)
    print(Q)



