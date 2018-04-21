#encoding:utf-8
import pandas as pd
import numpy as np
import math
#***************************************** getters ***********************************************************

def get_pre_by_country(df):
    grouped = df.groupby('country')
    size = grouped.size()
    sum = size.sum()
    pre = size/sum
    return pre.sort_values(ascending = False)

def get_paper_by_class(df_paper,country,ass):
    df_paper_country = df_paper[df_paper['country'] == country]
    if ass == 'CCF':
        df_paper_A = df_paper_country[df_paper_country['CCF_classification'] == 'A']
        df_paper_B = df_paper_country[df_paper_country['CCF_classification'] == 'B']
        df_paper_C = df_paper_country[df_paper_country['CCF_classification'] == 'C']
        return df_paper_A,df_paper_B,df_paper_C
    if ass == 'CORE':
        df_paper_AP = df_paper_country[df_paper_country['CORE_classification'] == 'A*']
        df_paper_A = df_paper_country[df_paper_country['CORE_classification'] == 'A']
        df_paper_B = df_paper_country[df_paper_country['CORE_classification'] == 'B']
        df_paper_C = df_paper_country[df_paper_country['CORE_classification'] == 'C']
        return df_paper_AP,df_paper_A, df_paper_B, df_paper_C
    else :
        return 0

def get_paper_pres_by_country(df_paper,country):
    if country == 'China':
        df_paper_chn_A, df_paper_chn_B, df_paper_chn_C = get_paper_by_class(df_paper, 'China', 'CCF')
        size_A = df_paper_chn_A.groupby('country')

def get_mean_paper_cnt_in_continent(df_paper,df_con):
    df_con_loc = df_con.copy()
    del df_con_loc['cnt_paper']
    df_con_loc = add_count_paper_to_cons_year(df_paper, df_con_loc)
    df_has_Continent_in_db = df_con_loc[df_con_loc['cnt_paper'].notnull()]
    grouped = df_has_Continent_in_db.groupby('Continent')
    return grouped['cnt_paper'].mean()

def pre_bycat(df_paper_has_cat):
    size = df_paper_has_cat.groupby('CCF_category').size()
    pre = size/size.sum()
    return pre

def pre_by_class(df_paper):
    size = df_paper.groupby('CCF_classification').size()
    pre = size/size.sum()
    return pre

def get_Npaper_per_con(df_paper_all,df_conference):
    li = ['1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0']
    grouped_paper_size = df_paper_all.groupby('CCF_category').size()
    grouped_conference_size = df_conference.groupby('CCF_category').size()
    pre = grouped_paper_size/grouped_conference_size
    return pre

def get_pre1(df_res):
    df_res.loc['Row_sum'] = df_res.apply(lambda x: x.sum())
    df_res = df_res.div(df_res.loc['Row_sum'])
    return df_res[:-1]

def get_in1_by_country(df,df_cons):
    df_aus = df[df['country'] == 'Australia']
    df_chn = df[df['country'] == 'China']
    df_us = df[df['country'] == 'United States']
    df1 = get_avg_papercnt_bycat(df,df_cons)
    df2 = get_avg_papercnt_bycat(df_aus,df_cons)
    df3 = get_avg_papercnt_bycat(df_chn,df_cons)
    df4 = get_avg_papercnt_bycat(df_us,df_cons)
    df_res = pd.concat([df1, df2, df3, df4], axis=1)
    df_res.columns = ['Total', 'Aus', 'China', 'US']
    return df_res

def get_avg_papercnt_bycat(df_paper,df_con):
    cons_size = df_con.groupby('CCF_category').size()
    paper_size = df_paper.groupby('CCF_category').size()
    avg = paper_size/cons_size
    return avg

def get_cons_all_cntPaper(df_paper,df_cons):
    cnt_paper = df_paper.groupby('con_id').size()
    df_cnt_paper = pd.DataFrame(cnt_paper, columns=['cnt_paper'])
    df_conference_with_cnt_paper = pd.merge(df_cons, df_cnt_paper, how='left', left_on='abbr', right_index = True)
    return df_conference_with_cnt_paper

def get_cons_year_cntPaper(df_paper,df_cons,year):
    cnt_paper = df_paper.groupby('con_id').size()
    cnt = 'cnt_paper' + str(year)
    df_cnt_paper = pd.DataFrame(cnt_paper, columns=[cnt])
    df_conference_with_cnt_paper = pd.merge(df_cons, df_cnt_paper, how='left', left_on='conference_id', right_index = True)
    return df_conference_with_cnt_paper

def get_ind(df_paper_all,df_con,year,cat,country,clas=-1):
    #year:int  cat,country: str
    if clas != -1:
        df_paper_all_tmp = df_paper_all[df_paper_all['CCF_classification']== clas]
    else:
        df_paper_all_tmp = df_paper_all
    df_paper_c_tmp = df_paper_all_tmp[df_paper_all_tmp['country'] == country]
    df_paper_c_tmp = df_paper_c_tmp[df_paper_c_tmp['publish_year'] == year]
    df_paper_all_tmp = df_paper_all_tmp[df_paper_all_tmp['publish_year'] == year]
    ind = 0
    df_con_cat = df_con[df_con['CCF_category'] == cat]#分类所有会议
    #todo 统计当年某会议的发文量
    if clas != -1:
        df_con_cat = df_con_cat[df_con_cat['CCF_classification'] == clas]
    cnt = 'cnt_paper' + str(year)
    pre = 'pre_paper' + str(year)
    df_con_with_year_cnt_paper = get_cons_year_cntPaper(df_paper_all_tmp, df_con_cat,year)
    df_con_with_year_cnt_paper[pre] = df_con_with_year_cnt_paper[cnt]/float(df_con_with_year_cnt_paper[cnt].sum())
    con_count = len(df_con_with_year_cnt_paper)
    cnt_country_year = len(df_paper_c_tmp) #sigma N(y,c,con)
    #print con_count
    if cnt_country_year == 0:
        return 0
    module_con_pre = 0
    module_country_pre = 0
    for ID in df_con_with_year_cnt_paper['conference_id']:
        cnt_country_con = len(df_paper_c_tmp[df_paper_c_tmp['con_id'] == ID]) # N(y,c,con)
        country_pre = cnt_country_con*1.0/cnt_country_year
        con_this = df_con_with_year_cnt_paper[df_con_with_year_cnt_paper['conference_id'] == ID]
        con_pre = float(con_this[pre])

        ind +=  country_pre *con_pre
        module_con_pre += con_pre*con_pre
        module_country_pre += country_pre*country_pre
        print 'current id is ' + ':' + str(ID)
        print 'con_pre_is' + ':' + str(con_pre)
        print 'country_pre_is' + ':' + str(country_pre)
        print 'ind' + ':' + str(ind)
        print 'module_con_pre' + ':' + str(module_con_pre)
        print 'module_country_pre' + ':' + str(module_country_pre)
        print '\n'

    ind = ind/math.sqrt(module_con_pre)#todo 处理0的情况
    ind = ind/math.sqrt(module_country_pre)
    return ind

def get_indicator(df_paper_all,df_con,year,cat,country,clas=-1):
    #year:int  cat,country: str
    if clas != -1:
        df_paper_all_tmp = df_paper_all[df_paper_all['CCF_classification']== clas]
    else:
        df_paper_all_tmp = df_paper_all
    df_paper_c_tmp = df_paper_all_tmp[df_paper_all_tmp['country'] == country]
    df_paper_c_tmp = df_paper_c_tmp[df_paper_c_tmp['publish_year'] == year]
    df_paper_all_tmp = df_paper_all_tmp[df_paper_all_tmp['publish_year'] == year]
    ind = 0
    df_con_cat = df_con[df_con['CCF_category'] == cat]#分类所有会议
    #todo 统计当年某会议的发文量
    if clas != -1:
        df_con_cat = df_con_cat[df_con_cat['CCF_classification'] == clas]
    cnt = 'cnt_paper' + str(year)
    pre = 'pre_paper' + str(year)
    df_con_with_year_cnt_paper = get_cons_year_cntPaper(df_paper_all_tmp, df_con_cat,year)
    df_con_with_year_cnt_paper[pre] = df_con_with_year_cnt_paper[cnt]/float(df_con_with_year_cnt_paper[cnt].sum())
    con_count = len(df_con_with_year_cnt_paper)
    cnt_country_year = len(df_paper_c_tmp) #sigma N(y,c,con)
   # print con_count
# 计算发文量最大的会议比例
    if cnt_country_year == 0:
        return 0
    module_con_pre = 0
    module_country_pre = 0
    pre_max = df_con_with_year_cnt_paper[pre].max()
    df_con_with_year_cnt_paper[pre] = df_con_with_year_cnt_paper[pre].fillna(0)
    for ID in df_con_with_year_cnt_paper['conference_id']:
        cnt_country_con = len(df_paper_c_tmp[df_paper_c_tmp['con_id'] == ID]) # N(y,c,con)
        country_pre = cnt_country_con*1.0/cnt_country_year
        con_this = df_con_with_year_cnt_paper[df_con_with_year_cnt_paper['conference_id'] == ID]
        con_pre = float(con_this[pre])

        ind +=  country_pre *con_pre/pre_max
        module_con_pre += con_pre*con_pre
        module_country_pre += country_pre*country_pre
        # print 'current id is ' + ':' + str(ID)
        # print 'con_pre_is' + ':' + str(con_pre)
        # print 'country_pre_is' + ':' + str(country_pre)
        # print 'ind' + ':' + str(ind)
        # print 'module_con_pre' + ':' + str(module_con_pre)
        # print 'module_country_pre' + ':' + str(module_country_pre)
        # print '\n'

    #ind = ind/math.sqrt(module_con_pre)#todo 处理0的情况
    #ind = ind/math.sqrt(module_country_pre)
    return ind

def get_indicator_year_cat_dic(df_paper, df_conference,cat,clas=-1):
    dic = {}
    for country in ['China', 'United States', 'Australia']:
        res = []
        for year in range(2005, 2017):
            ind = get_indicator(df_paper, df_conference, year, cat=cat, country=country, clas=clas)
            res.append(ind)
        dic[country] = res
    df = pd.DataFrame(dic)
    df.index = range(2005,2017)
    df.plot(title = cat + str(clas))
    return dic

def get_country_con_pre(df_paper_all,country,year,ID,all = -1):
    df_paper_all_tmp = df_paper_all
    df_paper_c_tmp = df_paper_all_tmp[df_paper_all_tmp['country'] == country]
    df_paper_c_tmp = df_paper_c_tmp[df_paper_c_tmp['publish_year'] == year]
    df_paper_all_tmp = df_paper_all_tmp[df_paper_all_tmp['publish_year'] == year]
    cnt_country_con = len(df_paper_c_tmp[df_paper_c_tmp['con_id'] == ID])
    cnt_world_con = len(df_paper_all_tmp[df_paper_all_tmp['con_id'] == ID])
    cnt_country_year =  len(df_paper_c_tmp)
    cnt_world_year =  len(df_paper_all_tmp)
    if cnt_world_con != 0 :
        pre_country_world = cnt_country_con*1.0/cnt_world_con
    else :
        pre_country_world = 0
        #print(country+'have pre_country_world  0 in year' + str(year) + 'in con' + str(ID))
    if cnt_world_year != 0:
        pre_con_world = cnt_world_con*1.0/cnt_world_year
    else :
        pre_con_world = 0
        #print('have pre_con_world 0 in year' + str(year) + 'in con' + str(ID))
    if cnt_country_year!=0:
        pre_con_in_country = cnt_country_con*1.0/cnt_country_year
    else:
        pre_con_in_country = 0
        #print(country + 'have pre_con_in_country  0 in year' + str(year) + 'in con' + str(ID))
    if all == -1:
        return pre_country_world,pre_con_world,pre_con_in_country
    return pre_country_world,pre_con_world,pre_con_in_country,cnt_world_con

def get_all_year_pres(df_paper,country,ID):
    li_pre_country_world = []
    li_pre_con_world = []
    li_pre_con_in_country = []
    for year in range(2005, 2017):
        pre_country_world, pre_con_world, pre_con_in_country = get_country_con_pre(df_paper, country, year, ID)
        li_pre_country_world.append(pre_country_world)
        li_pre_con_world.append(pre_con_world)
        li_pre_con_in_country.append(pre_con_in_country)
    return  li_pre_country_world,li_pre_con_world ,li_pre_con_in_country

def get_growth_rate1(df_paper,country):
    df_paper_tmp = df_paper[df_paper['country'] == country]
    df_paper_tmp['two'] = ((df_paper_tmp['publish_year'] + 1) / 2).astype(int) * 2
    size = df_paper_tmp.groupby('two').size()
    df_size = pd.DataFrame(size)
    df_size.columns = ['cnt_paper']
    df_rise = pd.DataFrame(np.array(df_size.cnt_paper[1:]) - np.array(df_size.cnt_paper[:-1]), columns=['rise'])
    df_rise.index = range(2008,2018,2)
    df = pd.merge(df_rise,df_size,left_index=True,right_index = True)
    df['rise_pro'] = df['rise'] / df['cnt_paper']
    df['rise_pro'] = df['rise']/df['cnt_paper']
    df.index = range(2008,2018,2)
    return df['rise_pro']

def get_growth_rate(df_paper,country):
    df_paper_tmp = df_paper[df_paper['country'] == country]
    size = df_paper_tmp.groupby('publish_year').size()
    df_size = pd.DataFrame(size)
    df_size.columns = ['cnt_paper']
    df_rise = pd.DataFrame(np.array(df_size.cnt_paper[1:]) - np.array(df_size.cnt_paper[:-1]), columns=['rise'])
    df_rise.index = range(2006,2016)
    df = pd.merge(df_rise,df_size,left_index=True,right_index = True)
    df['rise_pro'] = df['rise'] / df['cnt_paper']
    df['rise_pro'] = df['rise']/df['cnt_paper']
    df.index = range(2006,2016)
    return df['rise_pro']

def get_country_rise_dic(df_paper,country_list):
    dic = {}
    for country in country_list:
        dic[country] = get_growth_rate(df_paper,country)
    res = pd.DataFrame(dic)
    return res

def get_paper_in_active_cons(df_con,df_paper):
    dic = {}
    for id in df_con['conference_id']:
        dic[id] = 1
    df_paper['act'] = df_paper['con_id'].map(dic)
    df_tmp = df_paper.dropna()
    return df_tmp

def get_paper_in_import_cons(df_paper,df_cons):
    # define important cons by cnt_paper
    df_tmp = df_paper
    cons = df_cons[df_cons['CCF_classification'] == 'C'].sort_values(by='cnt_paper').tail(15)['conference_id']
    dic = {}
    for id in cons:
        dic[id] = 1
    df_tmp['act'] = df_tmp['con_id'].map(dic)
    df_tmp = df_tmp.dropna()
    return df_tmp

def get_df_paper_author_all(df_paper,df_a2p,df_author):
    df_author_tmp = df_author[[ u'author_id',u'index']]
    df_a2p_tmp = df_a2p[[u'paper_id', u'author_id', u'affiliation_id', u'order']]
    df_paper_tmp = df_paper[[ u'paper_id',
       u'publish_year', u'country', u'con_id', u'citation_count', u'co_state']]
    df_paper_author_all  = pd.merge(pd.merge(df_author_tmp,df_a2p_tmp,on = 'author_id'),df_paper_tmp,on = 'paper_id')
    return df_paper_author_all

#*****************************************  location traitement***********************************************************
def location_c2c_convert(df_locat,c2c_dic):
    res = {}
    i = 0
    for locat,index in zip(df_locat['location'],df_locat['index']):
        continent = check_location_in(locat,c2c_dic)
        res[index] = continent
        i += 1
        if i % 1000 == 0:
            print "%d finished." %i
    df_res = pd.Series(res)

    return df_res

def check_location_in(locat,c2c_dic):
    #used in location c2c convert
    for key in c2c_dic.keys():
        for country in c2c_dic[key]:
            if locat.lower().__contains__(country.lower()):
                return key
    return 'not found'

def get_c2c_dic(df_c2c):
    #get a dic whose key is continent and the values are countries in the continent
    grouped_country = df_c2c.groupby('continent')
    dic = {}
    for name,group in grouped_country:
        dic[name] = group
    c2c_dic = {}
    for key in dic.keys():
        c2c_dic[key] = dic[key]['country'].values
    return c2c_dic

def location_all_convert(df_locat,c2c_dic,cs2c_dic,s2c_dic):
    df_res  = location_c2c_convert(df_locat,c2c_dic)
    df_locat['Continent'] = df_res
    loc_not_found = df_locat[df_locat['Continent'] == 'not found']
    df_res1 = location_c2c_convert(loc_not_found,cs2c_dic)
    df_res1 = pd.DataFrame(df_res1)
    df_res1.columns = ['location']
    df_res1['index'] = df_res1.index
    df_res2 = location_c2c_convert(df_res1,c2c_dic)
    df_tmp = df_locat['Continent']
    df_1 = df_tmp.replace('not found', np.nan)
    df_1 = df_1.fillna(df_res2)
    df_locat['Continent'] = df_1

#*****************************************  group func ***********************************************************

def divsum(arr):
    return arr*1.0/arr.sum()

def func_by_country(func,df):
    df_aus = df[df['country'] == 'Australia']
    df_chn = df[df['country'] == 'China']
    df_us = df[df['country'] == 'United States']
    df1 = func(df)
    df2 = func(df_aus)
    df3 = func(df_chn)
    df4 = func(df_us)
    df_res = pd.concat([df1,df2,df3,df4],axis=1)
    df_res.columns = ['Total','Aus','China','US']
    return df_res

def func2_by_country(func,df,df_con):
    df_aus = df[df['country'] == 'Australia']
    df_chn = df[df['country'] == 'China']
    df_us = df[df['country'] == 'United States']
    df1 = func(df,df_con)
    df2 = func(df_aus,df_con)
    df3 = func(df_chn,df_con)
    df4 = func(df_us,df_con)
    df_res = pd.concat([df1,df2,df3,df4],axis=1)
    df_res.columns = ['Total','Aus','China','US']
    return df_res

def func_by_year(func,df,year_list,in_func = -1):
    list_df_res = []
    for year in year_list:
        df_year = df[df['publish_year'] == year]
        if in_func == -1:
            df_tmp = func(df_year)
        else:
            df_tmp = func(in_func,df_year)
        list_df_res.append(df_tmp)
    return list_df_res

def func_by_CCFclass(df,func):
    df_A = df[df['CCF_classification'] == 'A']
    df_B = df[df['CCF_classification'] == 'B']
    df_C = df[df['CCF_classification'] == 'C']
    df1 = func_by_country(func,df)
    df2 = func_by_country(func,df_A)
    df3 = func_by_country(func,df_B)
    df4 = func_by_country(func,df_C)
    df1 = df1.div(df1['Total'],axis = 0)
    df2 = df2.div(df2['Total'], axis=0)
    df3 = df3.div(df3['Total'], axis=0)
    df4 = df4.div(df4['Total'], axis=0)
    del df1['Total']
    del df2['Total']
    del df3['Total']
    del df4['Total']
    return df1,df2,df3,df4

def func_div_total(df_pre):
    df_res = df_pre.div(df_pre['Total'],axis = 0)
    del df_res['Total']
    return df_res

#*****************************************  group method ***********************************************************

def bycat(df_paper_has_cat):
    size = df_paper_has_cat.groupby('CCF_category').size()
    return size

def byyear(df_paper_has_year):
    size = df_paper_has_year.groupby('publish_year').size()
    return size

def byclass(df_paper):
    size = df_paper.groupby('CCF_classification').size()
    return size
#*****************************************  merge and add ***********************************************************

def add_count_paper_to_cons(df_paper_all,df_cons):
    cnt_paper = df_paper_all.groupby('con_id').size()
    df_cnt_paper = pd.DataFrame(cnt_paper, columns=['cnt_paper'])
    df_conference_with_cnt_paper = pd.merge(df_cons, df_cnt_paper, how='left', left_on='con_id', right_index=True)
    return df_conference_with_cnt_paper

def add_count_paper_to_cons_year(df_paper_all,df_cons):
    cnt_paper = df_paper_all.groupby('year_con').size()
    df_cnt_paper = pd.DataFrame(cnt_paper, columns=['cnt_paper'])
    df_conference_with_cnt_paper = pd.merge(df_cons, df_cnt_paper, how='left', left_on='abbr', right_index = True)
    return df_conference_with_cnt_paper

def merge_head_cons_paper(head_cons,df_paper):
    paper_list = []
    for con_id in head_cons:
        df_paper_tmp = df_paper[df_paper['con_id'] == con_id]
        paper_list.append(df_paper_tmp)
    df_res = pd.concat(paper_list,axis=0)
    return df_res

def add_author_number_to_paper(df_paper,df_a2p):
    cnt_author = df_a2p.groupby('paper_id').size()
    df_cnt_author = pd.DataFrame(cnt_author, columns=['cnt_author'])
    df_paper_with_cnt_author = pd.merge(df_paper, df_cnt_author, how='left', left_on='paper_id', right_index=True)
    return df_paper_with_cnt_author

def add_cite_year_to_relation(df_relation,df_paper):
    # 在relation中 src是被引， 因此判断年限应该用dst的年，即merge  dst和该paper的发表年
    df_paper_dst_year = df_paper[['paper_id','publish_year']]
    df_paper_dst_year.columns = ['dst_paper_id','dst_publish_year']
    df_relation_with_dst_year = pd.merge(df_relation,df_paper_dst_year,left_on='dst_id',right_on = 'dst_paper_id')
    df_paper_src_year = df_paper[['paper_id', 'publish_year']]
    df_paper_src_year.columns = ['src_paper_id','src_publish_year']
    df_relation_with_year = pd.merge(df_relation_with_dst_year,df_paper_src_year,left_on='src_id',right_on = 'src_paper_id')
    return df_relation_with_year

def add_collabarative_state(df_paper,df_a2p,df_aff):
    df_a2p_with_country = pd.merge(df_a2p, df_aff, on='affiliation_id', how='left')
    df_a2p_with_country = df_a2p_with_country[[u'paper_id', u'author_id', u'affiliation_id',
                                               u'order', u'affiliation_country']]

    dic = {}
    grouped = df_a2p_with_country.groupby(['paper_id'])
    for id, group in grouped:
        size = group.groupby('affiliation_country').size()
        if len(size) > 1:
            dic[id] = 1
        else:
            dic[id] = 0
    dic = pd.Series(dic)
    df_state = pd.DataFrame(dic)
    df_state.columns = ['co_state']
    df_state['paper_id'] = df_state.index
    res = pd.merge(df_paper,df_state,on = 'paper_id')
    return res
    #for group in grouped:

    #df_cnt_author = pd.DataFrame(cnt_author, columns=['cnt_author'])
    #df_paper_with_cnt_author = pd.merge(df_paper, df_cnt_author, how='left', left_on='paper_id', right_index=True)

def add_mean(df):
    mean = pd.DataFrame(df.mean(),columns=['mean'])
    res = pd.concat([df,mean.T])
    return res

def add_mean_cif(df_con,df_res):
    m = df_res.groupby('con_id')['cif'].sum().div(df_res.groupby('con_id')['cif'].size())
    dm = pd.DataFrame(m)
    df_tmp = pd.merge(df_con,dm,left_on = 'conference_id',right_index = True)
    return df_tmp




