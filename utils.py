#encoding:utf-8
import numpy as np

def save_csv(df,name,path_name):
    df.to_csv(path_name+'/'+name+'.csv',encoding='utf_8')

def Compute_geometric_mean(lis):
    # 计算列表list的几何均值
    n = len(lis)
    x = np.asarray(lis)
    res = np.exp(1.0 / n) * (np.log(x + 1).sum())  # 平滑
    return res

def Compute_arithmetic_mean(lis):
    # 计算列表list的几何均值
    n = len(lis)
    # x = np.asarray(lis)
    res = 1.0 / n * sum(lis)
    return res

def count_Nan(df_to_count,colomn_name):
    df_to_count[[colomn_name]] = df_to_count[[colomn_name]].fillna('X')
    size = df_to_count.groupby(colomn_name).size()
    keys = size.keys()
    if 'X' in keys:
        return size['X']
    return 'no value NaN in colomn ' + colomn_name