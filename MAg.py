#encoding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import MySQLdb
from getters import *
import numpy as np
import seaborn as sns
from scipy.interpolate import spline
import scipy.stats as st
from tools import *
sns.set_style("whitegrid")

def extract_list_df_year(list_df_res,year_list):
    li_aus =[]
    li_chi = []
    li_us = []
    li_total = []
    for df in list_df_res:
        li_aus.append(df['Aus'])
        li_chi.append(df['China'])
        li_us.append(df['US'])
        li_total.append(df['Total'])
    df_aus = pd.concat(li_aus, axis=1)
    df_aus.columns = year_list
    df_chi = pd.concat(li_chi, axis=1)
    df_chi.columns = year_list
    df_us = pd.concat(li_us, axis=1)
    df_us.columns = year_list
    df_total = pd.concat(li_total, axis=1)
    df_total.columns = year_list
    return df_aus,df_chi,df_us,df_total

def year_average_citation_count(df_paper):
    mean = df_paper.groupby('publish_year')['citation'].mean()
    return mean
#*****************************************  filter func ***********************************************************
def filter_df(df_paper_all,country=-1,year=-1,cat=-1,classification=-1):
    df_tmp = df_paper_all
    if country != -1:
        df_tmp = df_tmp[df_tmp['country'] == country]
    if year != -1:
        df_tmp = df_tmp[df_tmp['publish_year'] == year]
    if cat != -1:
        df_tmp = df_tmp[df_tmp['CCF_category'] == cat]
    if classification != -1:
        df_tmp = df_tmp[df_tmp['CCF_classification'] == classification]
    return df_tmp

def filter_country(df,country_list):
    df_list = []
    for country in country_list:
        df_list.append(filter_df(df,country))
    return df_list

def filter_df_by_year(df,column,maxY,minY ):
    df = df[df[column] >= minY]
    df = df[df[column] <= maxY]
    return df

def filter_active_con_paper(df_paper,df_act_con):
    dic = create_act_con_dic(df_act_con)
    df_tmp = df_paper
    df_tmp['act'] = df_tmp['con_id'].map(dic)
    df_res = df_tmp.dropna()
    return df_res


#*****************************************  draw func ***********************************************************
def draw_global_pre(size,global_size):
    pre = size/global_size
    pre = pre.dropna()
    pre.sort_values(ascending=False).plot(kind = 'bar')
    return pre['China'],pre['Austria']

def draw_paper_pre_trend(df_paper):
    pieces = dict(list(df_paper.groupby('publish_year')))
    li_chi = []
    li_aus = []
    li_us = []
    for key in pieces.keys():
        df_tmp = pieces[key]
        df = func_by_country(byclass, df_tmp)
        df = func_div_total(df)
        li_chi.append(df['China'])
        li_aus.append(df['Aus'])
        li_us.append(df['US'])
    year_list = pieces.keys()
    df_aus = pd.concat(li_aus, axis=1)
    df_aus.columns = year_list
    df_chi = pd.concat(li_chi, axis=1)
    df_chi.columns = year_list
    df_us = pd.concat(li_us, axis=1)
    df_us.columns = year_list
    df_A = pd.concat([df_aus._ixs(0), df_chi._ixs(0), df_us._ixs(0)], axis=1)
    df_B = pd.concat([df_aus._ixs(1), df_chi._ixs(1), df_us._ixs(1)], axis=1)
    df_C = pd.concat([df_aus._ixs(2), df_chi._ixs(2), df_us._ixs(2)], axis=1)
    df_A.columns = ['Aus','China','US']
    df_B.columns = ['Aus', 'China', 'US']
    df_C.columns = ['Aus', 'China', 'US']
    fig, axes = plt.subplots(3, 1)
    df_A.fillna(0).plot(ax=axes[0], title='A')
    df_B.fillna(0).plot(ax=axes[1], title='B')
    df_C.fillna(0).plot(ax=axes[2], title='C')

    return df_A,df_B,df_C

def draw_con_paper_pre_trend(df_paper):
    df_res = func_by_country(byyear, df_paper)
    df_res = func_div_total(df_res)
    df_res.fillna(0).plot()

def draw_head_publisher_country(df_paper):
    df_paper.groupby('country').size().sort_values(ascending=False).head().plot(kind = 'bar')

def draw_pre_citation(df_paper_with_citation):
    df_res = func_by_country(year_average_citation_count, df_paper_with_citation)
    df_res = func_div_total(df_res)
    df_res.plot()

def draw_yearly_papercnt(df_paper,title = None):
    size = df_paper.groupby('publish_year').size()
    size.plot(title = title)

def draw_paper_distribution(df_paper_with_citation,country):
    df_A = df_paper_with_citation[df_paper_with_citation['CCF_classification'] == 'A']
    df_B = df_paper_with_citation[df_paper_with_citation['CCF_classification'] == 'B']
    df_C = df_paper_with_citation[df_paper_with_citation['CCF_classification'] == 'C']
    df_X = df_paper_with_citation[df_paper_with_citation['CCF_classification'] == 'X']
    yearly_sizeA = df_A.groupby('publish_year').size()
    yearly_sizeB = df_B.groupby('publish_year').size()
    yearly_sizeC = df_C.groupby('publish_year').size()
    yearly_sizeX = df_X.groupby('publish_year').size()
    yearly_size_all = df_paper_with_citation.groupby('publish_year').size()
    preA = yearly_sizeA/yearly_size_all
    preB = yearly_sizeB/ yearly_size_all
    preC = yearly_sizeC / yearly_size_all
    preX = yearly_sizeX / yearly_size_all
    df_ccf_res = pd.concat([preA, preB, preC, preX], axis=1)
    df_ccf_res.plot.area(title='CCF situation for' + country)

def draw_by_country(draw,df):
    df_aus = df[df['country'] == 'Australia']
    df_chn = df[df['country'] == 'China']
    df_us = df[df['country'] == 'United States']
    draw(df,'Total')
    draw(df_aus,'Australia')
    draw(df_chn,'China')
    draw(df_us,'United States')

def draw_mean_paper_cnt(df_paper,df_con):
    df_con_loc = df_con.copy()
    del df_con_loc['cnt_paper']
    df_con_loc = add_count_paper_to_cons_year(df_paper, df_con_loc)
    df_has_Continent_in_db = df_con_loc[df_con_loc['cnt_paper'].notnull()]
    grouped = df_has_Continent_in_db.groupby('Continent')
    grouped['cnt_paper'].mean().plot(kind = 'bar')

def draw_with_legend(df_res,country_list):
    markers = ['-o','-v','->','-1','-s','-h']
    for country,marker in zip(country_list,markers):
        tmp = df_res[country]
        xnew = np.linspace(2006, 2016, 30)
        smooth = spline(tmp.index, tmp.values, xnew)
       # plt.plot(xnew, smooth, marker,label=country)
        plt.plot(tmp, marker, label=country)
    plt.legend(country_list)
    plt.xlim(2006, 2016)

def draw_class_disb(df_paper,country_list,type = 'CCF'):
    for country in country_list:
        df_tmp = df_paper[df_paper['country'] == country]
        if type == 'CCF':
            size = df_tmp.groupby(['CCF_classification','publish_year']).size()
            res = size.groupby('publish_year').transform(divsum)
            res.unstack().T.plot.area(title='CCF situation for ' + country)
        else:
            size = df_tmp.groupby(['CORE_classification', 'publish_year']).size()
            res = size.groupby('publish_year').transform(divsum)
            res.unstack().T.plot.area(title='CORE situation for ' + country)

def draw_corr_trend(dic):
    df = pd.DataFrame(dic)
    df.index = range(2005,2017)
    df.plot()

def draw_all_pres(df_paper,df_con,cat,clas,type = 'CCF'):
        col_clas = type + '_classification'
        if cat != -1:
            df_paper_cat = df_paper[df_paper['CCF_category'] == cat]
            df_con_cat = df_con[df_con['CCF_category'] == cat]
        else:
            df_paper_cat = df_paper
            df_con_cat = df_con
        df_paper_tmp = df_paper_cat
        df_con_tmp =df_con_cat
        if clas != -1:
            df_paper_tmp = df_paper_cat[df_paper_cat[col_clas] == clas]
            df_con_tmp = df_con_cat[df_con_cat[col_clas] == clas]
        Local,Global = get_corr_coef(df_paper_tmp,df_con_tmp)
        df1 = pd.DataFrame(Local)
        df1.index = range(2005, 2017)
        df1.plot(title = 'Local CCCI in cat'+ str(cat) + 'class' + str(clas))
        df2 = pd.DataFrame(Global)
        df2.index = range(2005, 2017)
        df2.plot(title = 'Global CCCI in cat'+ str(cat) + 'class' + str(clas))

def draw_country_3_pres(df_paper,country,ID):
    li1, li2, li3 = get_all_year_pres(df_paper, country, ID)
    dic = {}
    dic['pre_con_in_country'] = li3
    dic['pre_con_world'] = li2
    dic['pre_country_world'] = li1
    df = pd.DataFrame(dic)
    df.index = range(2005,2017)
    df.plot()

def regress_draw_by_year(df_paper_size):
    size = df_paper_size.groupby('publish_year').size()
    df_size = pd.DataFrame(size,columns = ['paper_count'])
    df_size['year'] = df_size.index
    sns.pairplot(df_size, x_vars='year', y_vars=[],size = 7, kind='reg')
    size.plot(ylim = (0,size.max()+100),)

def multi_count_draw(df_paper_A, df_con_A,df_paper_C, df_con_C,df_paper_X, df_con_X,
                     df_paper_B, df_con_B , country_list, year_list,head_count):
    resA = find_country_head_cons_avg_cntpaper_by_year(df_paper_A, df_con_A
                                                      , country_list, year_list,
                                                      head_count, pre=True)
    resA.plot(title = 'A class')
    resB = find_country_head_cons_avg_cntpaper_by_year(df_paper_B, df_con_B
                                                      , country_list, year_list,
                                                      head_count, pre=True)
    resB.plot(title = 'B class')
    resC = find_country_head_cons_avg_cntpaper_by_year(df_paper_C, df_con_C
                                                      , country_list, year_list,
                                                      head_count, pre=True)
    resC.plot(title = 'C class')
    resX = find_country_head_cons_avg_cntpaper_by_year(df_paper_X, df_con_X
                                                      , country_list, year_list,
                                                      head_count, pre=True)
    resX.plot(title = 'X class')

def draw_country_ccf_in_or_not(df_paper_nn_ccf,country_list):
    dic  = {}
    df_tmp = df_paper_nn_ccf
    for country in country_list:
        df_country = df_tmp[df_tmp['country'] == country]
        size = df_country.groupby(['publish_year', 'in_ccf']).size()
        size.unstack().plot(title = country)
        dic[country] = size
    return dic

def draw_con_yearly_cnt(df_paper,con_id):
    df_tmp = df_paper[df_paper['con_id'] == con_id]
    df_tmp.groupby('publish_year').size().plot()

def draw_analyse_con(df_paper_in_con):
    draw_yearly_papercnt(df_paper_in_con)
    draw_con_paper_pre_trend(df_paper_in_con)
    draw_head_publisher_country(df_paper_in_con)
    draw_pre_citation(df_paper_in_con)

def draw_country_paper_ratio(df_paper,country_list):
    size = df_paper.groupby('country').size()
    pre = size[country_list]/size.sum()
    plt.figure()
    pre.plot(kind = 'bar')

def draw_country_cat_trend(df_paper,country_list,cat = -1,clas = -1):
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    li = []
    for year in range(2005, 2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        total = len(df_paper_year)
        dic = {}
        for country in country_list:
            df_paper_country = df_paper_year[df_paper_year['country'] == country]
            Nc = len(df_paper_country)
            pre = Nc*1.0/total
            dic[country] = pre
        li.append(dic)
    df = pd.DataFrame(li)
    df.index = range(2005,2017)
    draw_with_legend(df,country_list)

def get_overall_papercnt(df_paper,country_list,cat = -1,clas = -1):
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    size = df_paper.groupby(['country', 'publish_year']).size()
    df = size[country_list].unstack().T
    df['total'] = df_paper_tmp.groupby('publish_year').size()
    df = df.T
    df['total'] = df.apply(sum,axis = 1)
    return df.T

def draw_country_growth(df_paper,country_list,cat = -1,clas = -1,combine2 = -1):
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    rates = {}
    for country in country_list:
        if combine2 == -1:
            res = get_growth_rate(df_paper_tmp, country)
        else :
            res = get_growth_rate1(df_paper_tmp, country)
        rates[country] = res
    df_rates = pd.DataFrame(rates)

    draw_with_legend(df_rates, country_list)


    return df_rates

def draw_GINI_trend(df_paper,country_list,cat = -1,clas = -1):
    res = []
    for year in range(2005, 2017):
        df_paper_year = df_paper[df_paper['publish_year'] == year]
        gini = calculate_GINI(df_paper_year,country_list,cat=cat,clas = clas)
        res.append(gini[0])
    df = pd.DataFrame(res)
    del df['line of equality']
    df.index = range(2005,2017)

    return df

#*****************************************  find func ***********************************************************

def find_con(df_conference,id):
    res = df_conference[df_conference['conference_id'] == id]
    return res

def find_country_dominate_con(df_paper,df_conference,country,head_count,pre):
    #此处的conference不能含有cnt_paper
    head_cons = []
    head_cons_id = []
    df_conference_with_cnt_paper = add_count_paper_to_cons(df_paper, df_conference)

    if pre == False:
        df_paper = df_paper[df_paper['country'] == country]
        size = df_paper.groupby('con_id').size()
        size = size[size>10]
        sorted_size = size.sort_values(ascending=False)
        for id in sorted_size.head(head_count).index:
            cur_con = df_conference_with_cnt_paper[df_conference_with_cnt_paper['con_id'] == id]
            head_cons.append(cur_con)
            head_cons_id.append(id)
        return head_cons,head_cons_id,0
    if pre == True:
        df_paper_chi = df_paper[df_paper['country'] == country]
        chi_size = df_paper_chi.groupby('con_id').size()
        chi_size = chi_size[chi_size > 5]
        all_size = df_paper.groupby('con_id').size()
        sorted_pre = (chi_size/all_size).sort_values(ascending=False)

        for id in sorted_pre.head(head_count).index:
            cur_con = df_conference_with_cnt_paper[df_conference_with_cnt_paper['con_id'] == id]
            head_cons.append(cur_con)
            head_cons_id.append(id)
        return head_cons,head_cons_id,sorted_pre.head(head_count)
    else :
        return head_cons,head_cons_id,0

def find_country_head_cons_avg_cntpaper(df_paper_recent,df_conference_without_cntpaper,country_list,
                                        head_count = 5,pre = True):
    res_list = []
    for country in country_list:
            head_cons, head_cons_id, sorted_pre = find_country_dominate_con(df_paper_recent,df_conference_without_cntpaper,country,head_count,pre)
            cnt_paper = 0
            for cons in head_cons:
                cnt_paper = cnt_paper + cons['cnt_paper'].values[0]
            avg_cnt = cnt_paper/len(head_cons)
            res_list.append(avg_cnt)
    return res_list

def find_country_head_cons_avg_cntpaper_by_year(df_paper_recent,df_conference_with_cntpaper,country_list,year_list,
                                                head_count = 5,pre = True):
    dic_df_res = {}
    for year in year_list:
        df_year = df_paper_recent[df_paper_recent['publish_year'] == year]
        res_list = find_country_head_cons_avg_cntpaper(df_year,df_conference_with_cntpaper,country_list,
                                        head_count,pre )
        dic_df_res[year] = res_list
        res = pd.DataFrame(dic_df_res)
        res.index = ['China', 'Australia', 'US']
    return res.T

def find_con_country_paper(df_paper,con_id,country):
    df_tmp = df_paper
    df_tmp = df_tmp[df_tmp['con_id'] == con_id]
    df_tmp = df_tmp[df_tmp['coutry'] == country]
    if len(df_tmp) != 0: return 1
    return 0

def find_publish_state(CCF_OR_CORE,df_con,df_paper,year = -1):
    if year != -1:
        df_tmp = df_paper[df_paper['publish_year'] == year]
    else:
        df_tmp = df_paper
    df_con_tmp = df_con
    res = {}
    if CCF_OR_CORE == 'CCF':
        for classification in ['A','B','C','X']:
            df_con_tmp1 = df_con_tmp[df_con_tmp['CCF_classification'] == classification]
            total = df_con_tmp1['conference_id'].size
            count = 0
            for id in df_con_tmp1['conference_id']:
                df_paper_in_con = df_tmp[df_tmp['con_id'] == id]
                if len(df_paper_in_con) != 0:
                    count+=1
            res[classification] = str(count) + '/' + str(total)
    if CCF_OR_CORE == 'CORE':
        for classification in ['A*','A','B','C','X']:
            df_con_tmp1 = df_con_tmp[df_con_tmp['CORE_classification'] == classification]
            total = df_con_tmp1['conference_id'].size
            count = 0
            for id in df_con_tmp1['conference_id']:
                df_paper_in_con = df_tmp[df_tmp['con_id'] == id]
                if len(df_paper_in_con) != 0:
                    count+=1
            res[classification] = str(count) + '/' + str(total)
    return res

def find_gap_cons(df_con,df_paper,para,gap):
    abn = []
    i = 0
    for id in df_con['conference_id']:
        li = []
        df_paper_con = df_paper[df_paper['con_id'] == id]
        size = df_paper_con.groupby('publish_year').size()
        for year in range(2005,2017):
            if year in size.index:
                li.append(size[year])
            else:
                li.append(0)

        j = 0
        parts = []
        while j<gap:
            s = 0
            i = 0
            while i+j<len(li):
                s += li[i+j]
                i += gap
            parts.append(s)
            j+=1
        #完成分组提取

        if max(parts)>sum(parts)*para:
            print "cur id" + str(id)
            print "cur li"
            print li
            abn.append(id)
        i+=1
        if i%100 == 0:
            print str(i) + " finished"

    return abn

def find_stop_cons(df_con,df_paper):
    abn = []
    i = 0
    for id in df_con['conference_id']:
        li = []
        df_paper_con = df_paper[df_paper['con_id'] == id]
        size = df_paper_con.groupby('publish_year').size()
        for year in range(2005,2017):
            if year in size.index:
                li.append(size[year])
            else:
                li.append(0)

        if sum(li[-3:-1])<10:
            print "cur id" + str(id)
            print "cur li"
            print li
            abn.append(id)
        i+=1
        if i%100 == 0:
            print str(i) + " finished"

    return abn

#*****************************************  prepare 4 RA ***********************************************************

def prepare_con_for_RA(df_paper,df_con_with_avg,country):
    df_con_tmp = df_con_with_avg
    df_paper_tmp = df_paper
    df_paper_tmp = df_paper_tmp[df_paper_tmp['country'] == country]
    record = []
    for year in range(2005,2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        for id in df_con_tmp['conference_id']:
            cnt_paper = len(df_paper_year[df_paper_year['con_id'] == id])
            if cnt_paper< 5 :
                continue
            con = df_con_tmp[df_con_tmp['conference_id'] == id]
            pre_country_world, pre_con_world, pre_con_in_country,cnt_world_con = get_country_con_pre(
                df_paper, country, year, id,0)
            res = {}
            res['year'] = year-2005
            res['cat'] = con.CCF_category.values[0]
            res['class'] = con.CCF_classification.values[0]
            res['con_avg'] = con.con_avg.values[0]
            res['con_year'] = cnt_world_con
            res['country_pub_paper'] = cnt_paper
            res['pre_country_world'] = pre_country_world
            res['pre_con_world'] = pre_con_world
            res['pre_con_in_country'] =  pre_con_in_country
            record.append(res)
    df = pd.DataFrame(record)
    return df

def covert_all_float(df_RA):
    df_4_RA = df_RA
    df_4_RA['CCF_category'] = df_4_RA['CCF_category'].replace('X',11)
    df_4_RA['CCF_category'] = df_4_RA['CCF_category'].astype(float).astype(int)
    df_4_RA['CCF_classification'] = df_4_RA['CCF_classification'].replace('A',3)
    df_4_RA['CCF_classification'] = df_4_RA['CCF_classification'].replace('B',2)
    df_4_RA['CCF_classification'] = df_4_RA['CCF_classification'].replace('C',1)
    df_4_RA['CCF_classification'] = df_4_RA['CCF_classification'].replace('X',0)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('A',3)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('B',2)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('C',1)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('X',0)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('A*', 4)
    df_4_RA['CORE_classification'] = df_4_RA['CORE_classification'].replace('SP', 0)
    return df_4_RA


def put_together(avg_paper_con,max, mean,CIF,CAI):
    df_avg_paper_con = avg_paper_con.T
    df_max = max.T
    df_mean = mean.T
    li = []
    for year in range(2010,2016):
        df_avg_paper_con_year = df_avg_paper_con[year]
        df_max_year = df_max[year]
        df_mean_year = df_mean[year]
        df_CIF_year = CIF[year]
        df_CAI_year = CAI[year-2007]
        df_year= pd.concat([df_avg_paper_con_year,df_max_year,df_mean_year,df_CIF_year,df_CAI_year],axis = 1)

        df_year.columns = [['avg_paper_con','max_H','avg_H','CIF',u'Aus_CAI',         u'Chi_CAI',       u'Ger_CAI',
                        u'Ind_CAI',u'Jap_CAI', u'US_CAI']]
        df_year['id'] = df_year.index
        df_year['year'] = [year] * len(df_year)

        li.append(df_year)
    res = pd.concat(li)
    df_con_ccf = df_con[[u'conference_id', u'CCF_classification', u'CCF_category']]
    df_res = pd.merge(res, df_con_ccf, left_index=True, right_on='conference_id')
    df_res = covert_all_float(df_res)
    return df_res

def prepare_RA(df_paper,country_list,df_con,df_a2p,df_author):
    df_avg_paper_con = calculate_avg_cnt_paper_con(df_con,df_paper,cnt = 3)
    df_max, df_mean = calculate_author_H_in_con(df_con, df_paper, df_a2p, df_author)
    df_CIF = calculate_CIF(df_con,df_relation,df_paper)
    li = calculate_CAI(df_paper, country_list, draw=-1)
    df_avg_paper_con = df_avg_paper_con.T
    df_avg_paper_con_2010 = df_avg_paper_con[2010]
    df_avg_paper_con_2015 = df_avg_paper_con[2015]
    df_max = df_max.T
    df_max_2010 = df_max[2010]
    df_max_2015 = df_max[2015]
    df_mean = df_mean.T
    df_mean_2010 = df_mean[2010]
    df_mean_2015 = df_mean[2015]

    df_CIF_2010 = df_CIF[2010]
    df_CIF_2015 = df_CIF[2015]

    df_CAI_2010 = li[3]
    df_CAI_2015 = li[-2]

    return 0
#*****************************************  abnormal analysis ***********************************************************

def get_con_active_year(df_con,df_paper):

    df_tmp = df_paper
    df_con_tmp = df_con
    res = {}
    for id in df_con_tmp['conference_id']:
        df_paper_in_con = df_tmp[df_tmp['con_id'] == id]
        size = df_paper_in_con.groupby('publish_year').size()
        active_year = len(size[size.values > 10])
        res[id] = active_year
    res = pd.Series(res)
    res = pd.DataFrame(res,columns=['active_year'])
    df_con_with_active_year = pd.merge(df_con_tmp,res,left_on='conference_id',right_index = True)
    df_con_with_active_year['con_avg'] = df_con_with_active_year['cnt_paper'].div(
        df_con_with_active_year['active_year'])
    df_con_with_active_year['con_avg'] = df_con_with_active_year['con_avg'].replace(float('inf'),0)
    df_con_with_active_year = df_con_with_active_year[df_con_with_active_year['con_avg'] != 0]

    return df_con_with_active_year

def check_abnormal_con(df_con,df_paper,para):
    #检查是否有异常发文量
    #para : define abnorm situation
    abn = []
    i = 0
    for id in df_con['conference_id']:
        li = []
        df_paper_con = df_paper[df_paper['con_id'] == id]
        size = df_paper_con.groupby('publish_year').size()
        for year in range(2005,2017):
            if year in size.index:
                li.append(size[year])
            else:
                li.append(0)
        li.sort()
        if li[-1]>sum(li)*para:
            print "cur id" + str(id)
            print "cur li"
            print li
            abn.append(id)
        i+=1
        if i%100 == 0:
            print str(i) + " finished"

    return abn

def remove_abnormal_con(df_con,abn):
    df_tmp = df_con
    for id in abn:
        df_tmp = df_tmp[df_tmp['conference_id']!= id]
    return df_tmp

def exclude_abn(df_con,df_paper,abn):
    df_con_tmp = remove_abnormal_con(df_con, abn)
    df_paper_tmp = filter_active_con_paper(df_paper, df_con_tmp)
    return df_con_tmp,df_paper_tmp

def create_act_con_dic(df_act_con):
    dic = {}
    for id in df_act_con['conference_id']:
        dic[id] = 1
    return dic

def in_sheet_calculate_chi_world(df_paper,classes,country):
    dicc = {}
    for clas in classes:
        li = []
        for new in [-1, 1]:
            d1 = {}
            Paa,Pac,Pah = calculate_chi_world(df_paper, clas, new, 10, country)
            d1['PAA'] = Paa
            d1['PAC'] = Pac
            d1['PAH'] = Pah
            li.append(d1)
        df = pd.DataFrame(li,index=['before 2013','after 2013'])
        dicc[clas] = df
    res = pd.DataFrame(dicc)
    return res
#*****************************************  calculators ***********************************************************

def calculate_author_H_index(df_paper_with_author):
    grouped = df_paper_with_author.groupby('author_id')
    dic = {}
    for name, group in grouped:
        res = group.sort_values(by='citation_count', ascending=False)['citation_count']
        res = pd.DataFrame(res)
        res.columns = ['citation_count']
        res['index'] = range(1, res['citation_count'].count() + 1)
        h_index = res[res['citation_count'] > res['index']]['index'].count()
        dic[name] = h_index
    return dic

def calculate_author_H_in_con(df_con,df_paper,df_a2p,df_author):
    df_tmp = df_paper[[ u'paper_id',
       u'publish_year', u'con_id']]
    df_a2p_tmp = df_a2p[[u'paper_id', u'author_id']]
    df_all = pd.merge(df_tmp,df_a2p_tmp,on = 'paper_id')
    df_cur = pd.merge(df_all,df_author,on = 'author_id')
    li_max = []
    li_mean = []
    i = 0
    for year in range(2010,2016):
        max_Hs = {}
        mean_Hs = {}
        df_year = df_cur[(df_cur['publish_year'] <= year)&(df_cur['publish_year'] >= year-2)]
        for id in df_con['conference_id']:
            # df_paper_in_con = df_tmp[(df_tmp['con_id'] == id)&(df_tmp['publish_year'] == year)][['paper_id','con_id']]
            # df_author_in_con = pd.merge(df_a2p,df_paper_in_con,on = 'paper_id').drop_duplicates('author_id')
            # df_author_in_con_H = pd.merge(df_author_in_con,df_author,on = 'author_id')
            df_in_con = df_year[df_year['con_id'] == id]['index']
            max_H = df_in_con.max()
            mean_H = df_in_con.mean()
            max_Hs[id] = max_H
            mean_Hs[id] = mean_H
            i+=1
            if i %100 == 0:
                print str(i) + 'finished'
        li_max.append(max_Hs)
        li_mean.append(mean_Hs)

    df_max = pd.DataFrame(li_max,index = range(2010,2016))
    df_mean = pd.DataFrame(li_mean, index=range(2010, 2016))
    return df_max,df_mean

def calculate_avg_cnt_paper_con(df_con,df_paper,cnt = 3):
    df_tmp = df_paper[ [u'paper_id',u'publish_year', u'con_id']]
    li = []
    for year in range(2010,2017):
        dic = {}
        df_year = df_tmp[(df_tmp['publish_year'] <= year-1)&(df_tmp['publish_year'] >= year-cnt)]

        for id in df_con['conference_id']:
            df_in_con = df_year[df_year['con_id'] == id]
            size = []
            for y in range(year-cnt+1,year+1):
                df_y = df_in_con[(df_in_con['publish_year'] == y)]
                size.append(len(df_y))
            sum_size = sum(size)
            active_year = 0
            for x in size:
                if x > 0.1*sum_size:
                    active_year += 1
            if active_year == 0:
                dic[id] = 0
            else:
                dic[id] = len(df_in_con)*1.0/active_year
        li.append(dic)
    df = pd.DataFrame(li, index=range(2010, 2017))
    return df

def calculate_CAI(df_paper,country_list,draw = -1):
    df_cur = df_paper
    li = []
    for year in range(2007, 2017):
        df_year = df_cur[(df_cur['publish_year'] <= year) & (df_cur['publish_year'] >= year - 2)]
        W0 = len(df_year)
        dic = {}
        for country in country_list:
            df_paper_country = df_year[df_year['country'] == country]
            C0 = len(df_paper_country)
            CAI = {}
            for id in df_con['conference_id']:
                df_paper_country_con = df_paper_country[df_paper_country['con_id'] == id]
                df_paper_con = df_year[df_year['con_id'] == id]
                Cc = len(df_paper_country_con)
                Wc = len(df_paper_con)
                if Cc != 0:
                    CAI[id] = (Cc*1.0/Wc)/(C0*1.0/W0) * 100
                else:
                    CAI[id] = 0
            dic[country] = CAI
        df = pd.DataFrame(dic)
        li.append(df)
    if draw != -1:
        return 'not implement'
    return li

def calculate_CIF(df_con,df_relation,df_paper):
    # conference Impact Factor
    dic = {}
    for year in range(2010,2017):
        df_paper_in_window = df_paper[(df_paper['publish_year'] <= year-3)&(df_paper['publish_year'] >= year-5)]
        df_relation_in_window = df_relation[(df_relation['dst_publish_year']<=year)&(df_relation['dst_publish_year']>=year-2)]
        for id in df_con['conference_id']:
            if id not in dic.keys():
                dic[id] = []
            df_paper_con_window = df_paper_in_window[df_paper_in_window['con_id'] == id]
            df_relation_relevant = pd.merge(df_paper_con_window,df_relation_in_window,left_on='paper_id'
                                            ,right_on='src_id')
            cnt_paper = len(df_paper_con_window)
            cnt_citation = len(df_relation_relevant)
            if cnt_paper == 0:
                dic[id].append(0)
            else:
                dic[id].append(cnt_citation*1.0/cnt_paper)
    df_IIF = pd.DataFrame(dic)
    df_IIF.index = range(2010,2017)
    df_IIF = df_IIF.T
    return df_IIF

def get_corr_coef(df_paper,df_con):
    dic1 = {}
    dic2 = {}
    for country in ['China','United States','Australia']:
        li_cov1 = []
        li_cov2 = []
        for year in range(2005,2017):
            li_pre_country_world = []
            li_pre_con_world = []
            li_pre_con_in_country = []
            for ID in df_con['conference_id']:
                #得到当年 每个会的三种比例
                pre_country_world, pre_con_world, pre_con_in_country = get_country_con_pre(df_paper, country, year, ID)
                li_pre_country_world.append(pre_country_world)
                li_pre_con_world.append(pre_con_world)
                li_pre_con_in_country.append(pre_con_in_country)
            arr1 = [li_pre_con_world,li_pre_con_in_country]
            arr2 = [li_pre_con_world,li_pre_country_world]
            cov_vector1 = np.corrcoef(arr1)
            cov_vector2 = np.corrcoef(arr2)
            cov1 = cov_vector1[0,1]
            cov2 = cov_vector2[0,1]
            li_cov1.append(cov1)
            li_cov2.append(cov2)
        dic1[country] = li_cov1
        dic2[country] = li_cov2
    return dic1,dic2

def calculate_TAI(df_paper,country_list,cat = -1,clas = -1,draw = -1):
    # Transformative Activity Index
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    W0 = len(df_paper_tmp)
    W0 = float(W0)
    size = df_paper_tmp.groupby('publish_year').size()
    Wi = []
    for year in range(2005, 2017):
        if year in size.index:
            Wi.append(size[year]/W0)
        else:
            Wi.append(0)

    dic = {}
    for country in country_list:
        df_paper_country = df_paper_tmp[df_paper_tmp['country'] == country]
        C0 = len(df_paper_country)
        size = df_paper_country.groupby('publish_year').size()
        TAI = []
        for year in range(2005, 2017):
            if year in size.index:
                TAI.append((size[year]/float(C0))/Wi[year-2005]*100)
            else:
                TAI.append(0)
        dic[country] = TAI
    df_tai = pd.DataFrame(dic)
    df_tai.index = range(2005,2017)
    if draw != -1:
        title = "TAI in cat " + str(cat) + " class " + str(clas)
        df_tai.plot(title = title)
    return df_tai

def calculate_CoAI(df_paper,country_list,divide_cnt_author,cat = -1,clas = -1):
    #Co-authorship Index
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    res = {}
    for year in range(2005,2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        N00 = len(df_paper_year)
        li = []
        for cnt_author in divide_cnt_author:
            if cnt_author == '>5':
                df_paper_part = df_paper_year[df_paper_year['cnt_author'] > 5]
            elif cnt_author == '3-4':
                df_paper_part = df_paper_year[(df_paper_year['cnt_author'] >2) &(df_paper_year['cnt_author'] <5) ]
            else:
                df_paper_part = df_paper_year[df_paper_year['cnt_author'] == cnt_author]
            N0j = len(df_paper_part)
            dic = {}
            for country in country_list:
                    df_paper_country = df_paper_year[df_paper_year['country'] == country]
                    Ni0 = len(df_paper_country)
                    df_paper_part_country = df_paper_part[df_paper_part['country'] == country]
                    Nij = len(df_paper_part_country)
                    CAI = (Nij*1.0/Ni0)/(N0j*1.0/N00) *100
                    dic[country] = CAI
            li.append(dic)
        df_year = pd.DataFrame(li)
        df_year.index = divide_cnt_author
        res[year] = df_year.T
    return res

def calculate_CC(df_paper,country_list,cat = -1,clas = -1):
    #Collaborative Coefficient
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    li = []
    for year in range(2005,2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        dic = {}
        for country in country_list:
            df_paper_country = df_paper_year[df_paper_year['country'] == country]
            N = len(df_paper_country)
            size = df_paper_country.groupby('cnt_author').size()
    # 1 - {sigma (1/j)Fj/N}
            CC = 1 - sum(((1.0/size.index) * size.values).values)/N
            J_avg = 1/(1-CC)
            dic[country] = J_avg
        li.append(dic)
    res = pd.DataFrame(li)
    res.index = range(2005,2017)
    return res

def calculate_IIF(df_con,df_relation,df_paper):
    # Inverse Impact Factor
    dic = {}
    for year in range(2005,2015):
        df_paper_year = df_paper[df_paper['publish_year'] == year]
        df_relation_in_window = df_relation[df_relation['dst_publish_year']<year+3]
        for id in df_con['conference_id']:
            if id not in dic.keys():
                dic[id] = []
            df_paper_con_year = df_paper_year[df_paper_year['con_id'] == id]
            df_relation_relevant = pd.merge(df_paper_con_year,df_relation_in_window,left_on='paper_id'
                                            ,right_on='src_id')
            cnt_paper = len(df_paper_con_year)
            cnt_citation = len(df_relation_relevant)
            if cnt_paper == 0:
                dic[id].append(0)
            else:
                dic[id].append(cnt_citation*1.0/cnt_paper)
    df_IIF = pd.DataFrame(dic)
    df_IIF.index = range(2005,2015)
    df_IIF = df_IIF.T
    return df_IIF

def calculate_PEI(df_con,df_paper,df_IIF,country_list,cat = -1,clas = -1,draw = -1):
    #Publication Effective Index (PEI).
    df_con_tmp = df_con
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_con_tmp = df_con_tmp[df_con_tmp['CCF_classification'] != 'X']
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_con_tmp = df_con_tmp[df_con_tmp['CCF_classification'] == clas]
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
        df_con_tmp = df_con_tmp[df_con_tmp['CCF_category'] == cat]
    TNIIFi = {}
    TNIIFi['Total'] = [0]*10
    TNP = {}
    TNP['Total'] = [0] * 10
    for country in country_list:
        TNIIFi[country] = [0]*10
        TNP[country] = [0]*10
    for year in range(2005,2015):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        TNP['Total'][year-2005] = len(df_paper_year)
        for id in df_con_tmp['conference_id']:
            df_paper_con = df_paper_year[df_paper_year['con_id'] == id]
            IIF_cur = df_IIF[year][id]
            TNIIFi['Total'][year-2005] += len(df_paper_con) * IIF_cur
            for country in country_list:
                df_paper_country = df_paper_con[df_paper_con['country'] == country]
                TNIIFi[country][year-2005]+= IIF_cur*len(df_paper_country)
                TNP[country][year-2005] += len(df_paper_country)
    df_TNIIF = pd.DataFrame(TNIIFi)
    df_TNP = pd.DataFrame(TNP)
    df_PEI = df_TNIIF.div(df_TNP)
    df_PEI = func_div_total(df_PEI)
    if draw!= -1:
        title = "PEI in cat "+ str(cat) + " class " + str(clas)
        df_PEI.plot(title = title)

    return df_PEI

def calculate_GINI(df_paper,country_list,cat = -1,clas = -1,draw = -1):
    #draw lorenz curve and calculate GINI
    Lorenz= {}
    gini = {}
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    for country in country_list:
        df_paper_country = df_paper_tmp[df_paper_tmp['country'] == country]
        size = df_paper_country.groupby('con_id').size().sort_values()
        size_all = df_paper_tmp.groupby('con_id').size().sort_values()
        pre = size/size_all
        pre = pre.fillna(0.000001)
        values = pre.values

        n = len(values)
        assert (n > 0), 'Empty list of values'
        sortedValues = sorted(values)
        cumm = [0]
        for i in range(n):
            cumm.append(sum(sortedValues[0:(i + 1)]))
        LorenzPoints = [[], []]
        sumYs = 0  # Some of all y values
        robinHoodIdx = -1  # Robin Hood index max(x_i, y_i)
        for i in range(1, n + 2):
            x = 100.0 * (i - 1) / n
            y = 100.0 * (cumm[i - 1] / float(cumm[n]))
            LorenzPoints[0].append(x)
            LorenzPoints[1].append(y)
            sumYs += y
            maxX_Y = x - y
            if maxX_Y > robinHoodIdx: robinHoodIdx = maxX_Y

        giniIdx = 100 + (100 - 2 * sumYs) / n  # Gini index
        gini[country] = giniIdx/100
        Lorenz[country] = LorenzPoints
    if draw != -1:
        plt.figure()

        for country in country_list:
            tmp = Lorenz[country]
            plt.plot(tmp[0], tmp[1],label=country)
        plt.plot([0, 100], [0, 100], '--',label = 'line of equality')
        tmp_country_list = country_list
        tmp_country_list.append('line of equality')
        plt.legend(tmp_country_list)
    return gini,Lorenz

def calculate_DCI(df_paper,country_list,dem = 0,cat = -1,clas = -1,draw = -1):
    # Domestic Collaborative Index
    # DCI = (Cd/C0)/(Wd/W0) *100
    # dem = 1 求出ICI  =0 求出DCI
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    res = []
    for year in range(2005, 2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        W0 = len(df_paper_year)
        W0 = float(W0)
        df_paper_dem = df_paper_year[df_paper_year['co_state'] == dem]
        Wd = len(df_paper_dem)
        dic = {}
        for country in country_list:
            df_paper_country = df_paper_year[df_paper_year['country'] == country]
            df_paper_dem_country = df_paper_dem[df_paper_dem['country'] == country]
            C0 = len(df_paper_country)
            Cd = len(df_paper_dem_country)
            XCI = (Cd*1.0/C0)/(Wd/W0) *100
            dic[country] = XCI
        res.append(dic)
    df = pd.DataFrame(res)
    df.index = range(2005,2017)
    if draw != -1:
        if dem == 0:
            title = "DCI in cat " + str(cat) + " class " + str(clas)
        else:
            title = "ICI in cat " + str(cat) + " class " + str(clas)
        df.plot(title = title)
    return df

def calculate_phc(df_paper,country_list,quant,cat = -1,clas = -1,draw = -1,relative = -1):
    # paper highly cited
    # phc% = #p(c>thres)/p(all)
    df_paper_tmp = df_paper
    if clas == 'not X':
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] != 'X']
    elif clas != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_classification'] == clas]
    if cat != -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['CCF_category'] == cat]
    res = []
    for year in range(2005, 2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        q = df_paper_year['citation_count'].quantile(quant)
        W0 = len(df_paper_year)
        df_paper_hc = df_paper_year[df_paper_year['citation_count'] >= q]
        hc_total = len(df_paper_hc)
        dic = {}
        for country in country_list:
            df_paper_country = df_paper_year[df_paper_year['country'] == country]
            df_paper_hc_country = df_paper_hc[df_paper_hc['country'] == country]
            C0 = len(df_paper_country)
            hc_country = len(df_paper_hc_country)
            if relative == -1:
                phc = hc_country*1.0/C0 *100
            else:
                phc = hc_country * 1.0 / hc_total *100
            dic[country] = phc
        res.append(dic)
    df = pd.DataFrame(res)
    df.index = range(2005,2017)
    if draw != -1:
        title = "phc% in cat " + str(cat) + " class " + str(clas)
        df.plot(title = title)
    return df

def calculate_all(df_paper,df_con,df_paper_author_all,df_CIF,df_avg_paper_con,df_loc,country_list,
                  active_thres = 20):
    # df_paper,df_con, 去除abn之后的
    # df_paper_author_all paper和author合并的df 包含所有作者的信息
    # df_CIF，df_avg_paper_con CIF，avg_paper_cnt表
    # active_thres 区别活跃会议的门槛值，不活跃会议视为未召开
    res = []
    df_paper_tmp = df_paper
    df_con_tmp = df_con
    df_paper_author_tmp = df_paper_author_all
    j = 0
    for year in range(2010,2017):
        df_paper_year = df_paper_tmp[df_paper_tmp['publish_year'] == year]
        df_paper_author_year = df_paper_author_tmp[df_paper_author_tmp['publish_year'] == year]
        df_loc_year = df_loc[df_loc['year'] == year]
        size = df_paper_year.groupby('con_id').size()
        active_con_ids = size[size>=active_thres].index.values
        con_num = len(size[size>=active_thres])
        df_paper_year_active = df_paper_year[df_paper_year['con_id'].isin(active_con_ids)]
        year_avg_con_paper = len(df_paper_year_active)*1.0/con_num
        dic = {}
        for id in active_con_ids:
            li = []
            df_author_con = df_paper_author_year[df_paper_author_year['con_id'] == id]
            df_paper_con = df_paper_year[df_paper_year['con_id'] == id]
            nor_coef = len(df_paper_con) *1.0 / year_avg_con_paper
            con_info = df_con_tmp[df_con_tmp['conference_id'] == id]
            # 需要的项为:
            # 计算类： avg_author_H,max_author_H,con_avg_author,con_cnt_paper,NPCs
            # 查找类： cif,con_avg_paper(近三年),clas,cat, cnt_paper
            # 附加类： year
            #还需预处理： loc
            avg_author_H = df_author_con['index'].mean()
            max_author_H = df_author_con['index'].max()
            con_avg_author = df_paper_con['cnt_author'].mean()
            con_cnt_paper = len(df_paper_con)
            #loc = df_loc[year,id]
            cif = df_CIF[year][id]
            con_avg_paper= df_avg_paper_con[id][year]
            clas = con_info['CCF_classification'].values[0]
            core = con_info['CORE_classification'].values[0]
            cat = con_info['CCF_category'].values[0]
            cnt_paper = con_info['cnt_paper'].values[0]
            li = li + [avg_author_H,max_author_H,con_avg_author,con_cnt_paper,cif,con_avg_paper,clas,core,cat, cnt_paper,year]
            #NPCs
            size_countries = df_paper_con.groupby('country').size()
            doms = [0]*6
            i = 0
            record = df_loc_year[df_loc_year['conference_id'] == id]
            for country in country_list:
                if country in size_countries.index:
                    li.append(size_countries[country]*1.0/nor_coef)
                else :
                    li.append(0)
            if len(record) != 0:
                i = 0
                loc = record['country'].values[0]
                for country in country_list:
                    if country in loc:
                        doms[i] = 1
                        break
                    if 'USA' in loc:
                        doms[1] = 1
                        break
                    i+=1
            li = li + doms
            dic[id] = li
            j+=1
            if j%200 == 0:
                print str(j) + " finished"
        df_year = pd.DataFrame(dic)
        df_year.index = ['avg_author_H','max_author_H','con_avg_author','con_cnt_paper','cif',
                  'con_avg_paper','CCF_classification','CORE_classification','CCF_category', 'cnt_paper','year','Chi_NPC','US_NPC'
                            ,'Aus_NPC','Ind_NPC','Ger_NPC','Jap_NPC','Chi_dom','US_dom'
                            ,'Aus_dom','Ind_dom','Ger_dom','Jap_dom']
        df_year = df_year.T
        res.append(df_year)
    df_res = pd.concat(res)
    df_res['con_id'] = df_res.index
    df_res_RA = covert_all_float(df_res)
    return df_res_RA

def calculate_gmeans(df_stata):
    grouped = df_stata.groupby(['new','ccf_classification','ccf_category'])
    res = []
    for (k1,k2,k3),group in grouped:
        gmeans = st.gmean(group[[u'chi_npc', u'us_npc', u'aus_npc', u'ind_npc', u'ger_npc', u'jap_npc']].apply(lambda x:x+1))
        li = list(gmeans)
        li.append(k1)
        li.append(k2)
        li.append(k3)
        li.append(-1)
        res.append(li)
    return res

def calculate_chi_world(df_paper,clas = -1,new = 0,order = 10,country = 'China'):
    #分13年以前和以后  不同分级中 中国作者平均H/世界平均H，
        # 中国CITATION/世界CITATION,中国MAX_H, 中国平均作者数/世界平均作者数
    #df_paper为所有paper记录，包含所有作者信息，用于计算作者贡献
    #df_paper_1为所有文章1作信息，用于计算文章贡献
    #中国CITATION/世界CITATION
    if clas != -1:
        df_paper_tmp = df_paper[df_paper['CCF_classification'] == clas]
    else:
        df_paper_tmp = df_paper
    if new == -1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['publish_year']<2013]
    elif new == 1:
        df_paper_tmp = df_paper_tmp[df_paper_tmp['publish_year']>2013]
    elif new != 0:
        return 0,0,0,0
    df_paper_1_tmp = df_paper_tmp[df_paper_tmp['order'] == 1]

    df_paper_1chi = df_paper_1_tmp[df_paper_1_tmp['country'] == country]
    df_paper_chi = df_paper_tmp[df_paper_tmp['country'] == country]
    #PAC: paper average citation
    PAC = df_paper_1_tmp['citation_count'].sum()*1.0/df_paper_1_tmp['citation_count'].size
    PAC_chi = df_paper_1chi['citation_count'].sum() * 1.0 / df_paper_1chi['citation_count'].size
    Pre_PAC = PAC_chi * 100.0/PAC
    #PAA: paper average author
    PAA = df_paper_tmp.size*1.0/df_paper_1_tmp.size
    PAA_chi = df_paper_chi.size*1.0/df_paper_1chi.size
    Pre_PAA = PAA_chi *100.0 /PAA
    #PAH: paper average H-index
    tmp = df_paper_tmp[df_paper_tmp['order']<order]
    chi_tmp = tmp[tmp['country'] == country]
    PAH = tmp['index'].sum()*1.0/tmp['index'].size
    PAH_chi = chi_tmp['index'].sum()*1.0/chi_tmp['index'].size
    Pre_PAH = PAH_chi *100.0 /PAH
    #max_H: max_h in this classification
    max_H = df_paper_chi['index'].max()
    #PEI： publication effeciency index
    #PEI = (chi_citation/world_citation)/(chi_paper_count/world_paper_count)
    PEI = (df_paper_1chi['citation_count'].sum()*100.0/df_paper_1_tmp['citation_count'].sum()
           )/((df_paper_1chi['citation_count'].size)*1.0/(df_paper_1_tmp['citation_count'].size))
    return round(Pre_PAA,2),round(Pre_PAC,2),round(Pre_PAH,2),round(PEI,2)


if __name__ == '__main__':
# *****************************************  DB conn ***********************************************************

    sql_ip = "localhost"  # 数据库地址
    port = 3306  # 数据库端口号
    user = "root"
    passwd = "ljl951128"
    db = "aminer_gai"
    conn = MySQLdb.connect(host=sql_ip, user=user, port=port, passwd=passwd, db=db, charset="utf8")
#*****************************************  read csv ***********************************************************

  #  df_Affiliation = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_aff.csv')
    df_aff = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_aff.csv')
    df_a2p = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_a2p.csv')
#    df_author = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_author.csv')
#    df_conference = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_conference.csv')
#    df_paper = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_paper_with_country.csv')
    df_relation = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_relationship.csv')

    df_author = pd.read_csv('D:\Projects\PythonProjects\csv\cur_use\df_author.csv')
    df_paper = pd.read_csv('D:\Projects\PythonProjects\csv/result\df_paper_more.csv')
    df_con = pd.read_csv('D:\Projects\PythonProjects\csv/result\df_con_more.csv')
    df_location = pd.read_csv('D:\Projects\PythonProjects\csv/new_records\df_series.csv')
    df_paper_author_all = pd.read_csv('D:\Projects\PythonProjects\csv\cur_use/df_paper_author_all.csv')

#*****************************************  bacis lists ***********************************************************

    cats = ['1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0']
    classes = ['A','B','C',-1,'X','not X']
    country_list = ['China','United States','Australia','India','Germany']
    divide_cnt_author = [1,2,'3-4','>5']
#*****************************************  simple traitements ***********************************************************

    # df_paper_with_class = pd.merge(df_paper_withcountry,df_conference,on = 'con_id')
    # df_paper_with_class = df_paper_with_class[['paper_id', 'publish_year', 'country',
    #                                            'con_id',
    #                                            'CCF_classification', 'CCF_category', 'CORE_classification']]
    # df_paper_with_class['country'] = df_paper_with_class['country'].fillna('Other')

#*****************************************  Analysis ***********************************************************

    # df_TAI = calculate_TAI(df_paper, country_list, cat=-1, clas=-1, draw=-1)
    #
    # df_paper = add_author_number_to_paper(df_paper,df_a2p)
    # df_CAI = calculate_CAI(df_paper,country_list,divide_cnt_author,cat = -1,clas = -1)
    #
    # df_CC = calculate_CC(df_paper,country_list,cat = -1,clas = -1)
    #
    df_relation = add_cite_year_to_relation(df_relation,df_paper)
    # df_IIF = calculate_IIF(df_con,df_relation,df_paper)#计算很慢
    #
    # df_PEI = calculate_PEI(df_con,df_paper,df_IIF,country_list,cat = -1,clas = -1,draw = -1)
    # gini,lorenz = calculate_GINI(df_paper,country_list,cat = -1,clas = -1,draw = -1)
    #
    # df_a2p_with_country = pd.merge(df_a2p,df_aff,on = 'affiliation_id',how = 'left')

    abn = find_stop_cons(df_con,df_paper)
    df_con,df_paper = exclude_abn(df_con,df_paper,abn)