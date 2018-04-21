#encoding:utf-8
import pandas as pd

def get_paper(conn):
    sql_paper = "SELECT paper_id, paper_title, paper_publicationYear, paper_nbCitation, paper_label, venue_venue_id FROM paper"
    df_paper = pd.read_sql(sql_paper, conn)
    return df_paper

def get_a2p(conn):
    sql_a2p = "SELECT * FROM a2p"
    df_a2p = pd.read_sql(sql_a2p, conn)
    return df_a2p

def get_author(conn):
    sql_author = "SELECT * FROM author"
    df_author = pd.read_sql(sql_author, conn)
    return df_author

def get_relationship(conn):
    sql_relationship = "SELECT relationship_src, relationship_dst FROM relationship"
    df_relationship = pd.read_sql(sql_relationship, conn)
    return df_relationship

def get_venue(conn):
    sql_venue = "SELECT * FROM venue"
    df_venue = pd.read_sql(sql_venue, conn)
    return df_venue

def get_dblp2ccf(conn):
    sql_dblp2ccf = "SELECT * FROM dblp2ccf"
    df_dblp2ccf = pd.read_sql(sql_dblp2ccf, conn)
    return df_dblp2ccf

def get_ccf(conn):
    sql_ccf = "SELECT * FROM ccf"
    df_ccf = pd.read_sql(sql_ccf, conn)
    return df_ccf

def get_dblp2core(conn):
    sql_dblp2core = "SELECT * FROM dblp2core"
    df_dblp2core = pd.read_sql(sql_dblp2core, conn)
    return df_dblp2core

def get_core(conn):
    sql_core = "SELECT * FROM core"
    df_core = pd.read_sql(sql_core, conn)
    return df_core

def get_ccf_with_dblp(df_ccf,df_dblp2ccf):
    return pd.merge(df_ccf, df_dblp2ccf, left_on='CCF_id', right_on='ccf_CCF_id')

def get_core_with_dblp(df_core,df_dblp2core):

    return  pd.merge(df_core, df_dblp2core, left_on='CORE_id', right_on='core_CORE_id')

def generate_relationship_paper(df_paper, df_a2p, df_author, df_relationship, df_venue, df_dblp2ccf, df_ccf):
	#根据数据库信息， 返回df_relationship，其中包含源和目的的发表年份、标签、最大H因子、第一作者所属国家
    #同时得到扩充的df_paper，包含venue以及ccf,dblp的信息
    #*********************************************此部分得到论文的最大H因子********************************************************
    #合并, 内连接为了得到每个论文作者的最大H因子
    df = pd.merge(df_paper, pd.merge(df_author, df_a2p, left_on='author_id', right_on='author_author_id'), left_on='paper_id', right_on='paper_paper_id')

    group_paper = df.groupby(df['paper_id'])
    #得到paper的最大H因子
    res_Hindex = group_paper.author_H_Index.max()
    #计数，得到最大H因子的个数

    #找到作者最大H因子，写到df_paper中
    df_Hindex = pd.DataFrame(res_Hindex)
    df_Hindex['paper_id'] = df_Hindex.index

    #*************************************此部分得到论文的J/C、computercategory***************************************************
    df2_part1 = pd.merge(df_paper, df_venue, left_on='venue_venue_id', right_on='venue_id')
    df2_part2 = pd.merge(df_ccf, df_dblp2ccf, left_on='CCF_id', right_on='ccf_CCF_id')
    df2_part2 = df2_part2[df2_part2['dblp_dblp_id']<999999999]
    df2 = pd.merge(df2_part1, df2_part2, left_on='dblp_dblp_id', right_on='dblp_dblp_id')
    df2 = df2[['paper_id', 'CCF_type', 'computercategory_computerCategory_id','CCF_id']]
    df2 = df2.drop_duplicates(['paper_id']) #去掉重复的paper_id数据

    df_paper = pd.merge(df_paper, df_Hindex, left_on='paper_id', right_on='paper_id', how='outer') #在paper表上加上Max H因子
    df_paper = pd.merge(df_paper, df2, left_on='paper_id', right_on='paper_id', how='outer') #在paper表上加上J/C, computer类别

    #找到每篇论文第一作者的国籍，写入df_paper中
    df_sub = df[df['a2p_order']==1]
    df_sub2 = df_sub[['paper_id', 'author_affiliation_name']]
    df_sub_China = df_sub2[df_sub2['author_affiliation_name'].str.contains('China')].copy()
    df_sub_Australia = df_sub2[df_sub2['author_affiliation_name'].str.contains('Australia')].copy()

    df_sub_China['country'] = 'China'
    df_sub_Australia['country'] = 'Australia'
    df_country = pd.concat([df_sub_China, df_sub_Australia])
    df_country = df_country.drop_duplicates(['paper_id']) #去除2446篇第一作者来自两个国家的情况，这里将澳大利亚的情况去掉

    #df_paper_inter = pd.merge(df_paper, df_country) #只包括中国和澳大利亚作者
    df_paper_outer = pd.merge(df_paper, df_country, how = 'outer')#包括所有国家作者，但是大部分国籍为NAN(除中澳外的其他国家)

    df_relationship = pd.merge(df_relationship, df_paper_outer[['paper_id','paper_publicationYear','paper_label','author_H_Index', 'CCF_type', 'computercategory_computerCategory_id', 'country']], left_on='relationship_src', right_on='paper_id')
    del df_relationship['paper_id']
    #换列名
    df_relationship.columns = ['relationship_src', 'relationship_dst', 'relationship_src_publicationYear', 'relationship_src_label', 'relationship_src_maxHindex', 'relationship_src_type', 'relationship_src_computerCategory', 'relationship_src_country']

    df_relationship = pd.merge(df_relationship, df_paper_outer[['paper_id','paper_publicationYear','paper_label','author_H_Index', 'CCF_type', 'computercategory_computerCategory_id', 'country']], left_on='relationship_dst', right_on='paper_id')
    del df_relationship['paper_id']
    df_relationship.columns = ['relationship_src', 'relationship_dst', 'relationship_src_publicationYear', 'relationship_src_label',
                               'relationship_src_maxHindex', 'relationship_src_type', 'relationship_src_computerCategory',
                               'relationship_src_country', 'relationship_dst_publicationYear', 'relationship_dst_label',
                               'relationship_dst_maxHindex', 'relationship_dst_type', 'relationship_dst_computerCategory',
                               'relationship_dst_country']
    #此时的df_relationship包含源和目的的发表年份、标签、最大H因子、type、 CCF类别、第一作者所属国家
    return df_paper_outer, df_relationship

def get_all(conn):
    df_paper = get_paper(conn)
    df_a2p = get_a2p(conn)
    df_author = get_author(conn)
    df_relationship = get_relationship(conn)
    df_venue = get_venue(conn)
    df_dblp2ccf = get_dblp2ccf(conn)
    df_ccf = get_ccf(conn)
    df_core = get_core(conn)
    df_dblp2core = get_dblp2core(conn)

    return df_paper, df_a2p, df_author, df_relationship, df_venue, df_dblp2ccf, df_ccf,df_core,df_dblp2core

def get_overlap(df_relationship):
	df_tmp = df_relationship[(df_relationship['relationship_src_label'].notnull()) & (df_relationship['relationship_dst_label'].notnull())]
	return df_tmp

def get_table(df_relationship, src_publicationYear, dst_publicationYear, src_country, dst_country):
    """
    params:
    src_publicationYear : 源论文集的发表年份 (-1表示不限制)
    dst_publicationYear : 引用论文的发表截止年份（从源论文发表的年份直到目的年份） (-1表示不限制)
    src_country : 源论文集的国家 （'NULL'表示不限制）
    dst_country : 目的论文集的国家 （'NULL'表示不限制）
    """
    # 只考虑CCF和CORE交叉部分的情况目前，因此先处理df_relationship
    df_tmp = get_overlap(df_relationship)
    if src_publicationYear != -1:
        # 源的发表年份受限，应按发表年份删选
        df_tmp = df_tmp[df_tmp['relationship_src_publicationYear'] == src_publicationYear]
    if dst_publicationYear != -1:
        # 目的截止年份受限
        df_tmp = df_tmp[df_tmp['relationship_dst_publicationYear'] <= dst_publicationYear]
    if src_country != 'NULL':
        df_tmp = df_tmp[df_tmp['relationship_src_country'] == src_country]
    if dst_country != 'NULL':
        df_tmp = df_tmp[df_tmp['relationship_src_country'] == dst_country]

    df_tmp = df_tmp[
        ['relationship_src', 'relationship_src_label', 'relationship_dst', 'relationship_dst_label']]  # 只要标签个数统计

    # 获取发表年份为2000年，且源和目的的label都存在的所有relationship，不考虑国家
    # df_2000_withLabel_withoutCountry = df_relationship[(df_relationship['relationship_src_publicationYear'] == 2000) & (df_relationship['relationship_src_label'].notnull()) & (df_relationship['relationship_dst_label'].notnull())]
    # df_2000_withLabel_withoutCountry = df_2000_withLabel_withoutCountry[['relationship_src', 'relationship_src_label', 'relationship_dst', 'relationship_dst_label']]

    # 按源和目的的label聚合
    grouped = df_tmp.groupby(['relationship_src_label', 'relationship_dst_label'])
    result = grouped.count().unstack().fillna(0)  # 得到表格结果，列表示源的标签，行表示目的的标签，值表示引用量
    return result

def get_table2(df_relationship):
    # 得到2000年、源是computer类别是8（人工智能）、源和目的都是Conference的字表视图
    df_relationship = get_overlap(df_relationship)
    # df_relationship[ (df_relationship['relationship_src_publicationYear'] == 2000) & (df_relationship['relationship_src_type'] == 'conference') & (df_relationship['relationship_dst_type'] == 'conference') & (df_relationship['relationship_src_computerCategory'] == 8)]
    df_relationship_cur = df_relationship[df_relationship['relationship_src_publicationYear'] == 2000]
    df_relationship_cur = df_relationship_cur[df_relationship_cur['relationship_src_label'] == 'A,A*']
    df_relationship_cur = df_relationship_cur[df_relationship_cur['relationship_src_type'] == 'conference']
    df_relationship_cur = df_relationship_cur[df_relationship_cur['relationship_dst_type'] == 'conference']
    df_relationship_cur = df_relationship_cur[df_relationship_cur['relationship_src_computerCategory'] == 8]

def get_ccf_core_combination(df_ccf,df_dblp2ccf,df_core,df_dblp2core):
    df_ccf_dblp = pd.merge(df_ccf,df_dblp2ccf,left_on='CCF_id',right_on = 'ccf_CCF_id')
    df_ccf_dblp = df_ccf_dblp.drop_duplicates('dblp_dblp_id')
    df_ccf_dblp = df_ccf_dblp [df_ccf_dblp ['dblp_dblp_id'] < 999999999]
    df_core_dblp = pd.merge(df_core,df_dblp2core,left_on='CORE_id',right_on = 'core_CORE_id')
    df_core_dblp = df_core_dblp.drop_duplicates('dblp_dblp_id')
    df_core_dblp = df_core_dblp[df_core_dblp['dblp_dblp_id'] < 999999999]
    df_combination = pd.merge(df_ccf_dblp,df_core_dblp,on = 'dblp_dblp_id',how = 'outer')
    df_combination = df_combination[['dblp_dblp_id', 'CCF_id', 'CORE_id', 'CCF_dblpname', 'CORE_dblpname',
                 'computercategory_computerCategory_id','CCF_type', 'CCF_classification', 'CORE_classification']]
    df_combination.columns = ['dblp_id', 'CCF_id', 'CORE_id', 'CCF_dblpname', 'CORE_dblpname', 'computerCategory',
                    'type','CCF_classification', 'CORE_classification']
    return df_combination

def merge_paper_venue(df_paper,df_venue):
    return pd.merge(df_paper,df_venue,left_on='venue_venue_id',right_on='venue_id')

def get_paper_all_label(df_paper_venue,df_cross):
    df_paper_all_label = pd.merge(df_paper_venue, df_cross, left_on='dblp_dblp_id', right_on='dblp_id', how='left')
    df_paper_all_label['label'] = df_paper_all_label['label'].fillna('Not in db')
    df_paper_all_label = df_paper_all_label[['paper_id', 'paper_title', 'paper_publicationYear',
       'paper_nbCitation', 'venue_venue_id',
       'author_H_Index', 'CCF_type', 'computercategory_computerCategory_id',
       'CCF_id_x', 'author_affiliation_name', 'country', 'dblp_id','CORE_id',
       'label','CCF_classification','CORE_classification']]
    df_paper_all_label.columns = ['paper_id', 'paper_title', 'paper_publicationYear',
       'paper_nbCitation', 'venue_venue_id',
       'author_H_Index', 'CCF_type', 'computercategory_computerCategory_id',
       'CCF_id', 'author_affiliation_name', 'country', 'dblp_id','CORE_id',
       'paper_label','CCF_classification','CORE_classification']
    return df_paper_all_label

def get_venue_with_all_labels(df_cross):
    df_cross['CORE_classification'] = df_cross['CORE_classification'].replace(['Australasian', 'L', 'National', 'Not ranked', 'Unranked'], "SP")
    df_cross['CCF_classification'] = df_cross['CCF_classification'].fillna('X')
    df_cross['CORE_classification'] = df_cross['CORE_classification'].fillna('X')
    df_cross['label'] = df_cross['CCF_classification'] + ',' + df_cross['CORE_classification']
    return df_cross

def get_paper_all_label_ori(df_paper,df_venue,df_ccf,df_dblp2ccf,df_core,df_dblp2core):
    df_paper_venue = merge_paper_venue(df_paper,df_venue)
    df_cross = get_ccf_core_combination(df_ccf,df_dblp2ccf,df_core,df_dblp2core)
    df_cross = get_venue_with_all_labels(df_cross)
    res = get_paper_all_label(df_paper_venue, df_cross)
    return res

def draw_crosstab_overlap(df_ccf,df_dblp2ccf,df_core,df_dblp2core):
    df_cross = get_ccf_core_combination(df_ccf, df_dblp2ccf, df_core, df_dblp2core)
    df_cross['CORE_classification'] = df_cross['CORE_classification'].replace(
        ['Australasian', 'L', 'National', 'Not ranked', 'Unranked'], "SP")
    df_cross['CCF_classification'] = df_cross['CCF_classification'].fillna('X')
    df_cross['CORE_classification'] = df_cross['CORE_classification'].fillna('X')
    crosstab = pd.crosstab(df_cross.CCF_classification, df_cross.CORE_classification, \
            margins=True)[['A*', 'A', 'B', 'C', 'SP','X', 'All']]
    crosstab = crosstab.div(crosstab['All'],axis = 0)
    df_to_draw = crosstab[['A*', 'A', 'B', 'C']].iloc[[0,1,2]]
    df_to_draw.plot(kind='bar', stacked=True)
    return crosstab

def draw_crosstab_overlap_paper(df_paper_cmp,df_venue,df_ccf,df_dblp2ccf,df_core,df_dblp2core):
    df_cross = get_paper_all_label_ori(df_paper_cmp, df_venue, df_ccf, df_dblp2ccf, df_core, df_dblp2core)
    df_cross['CORE_classification'] = df_cross['CORE_classification'].replace(
        ['Australasian', 'L', 'National', 'Not ranked', 'Unranked'], "SP")
    df_cross['CCF_classification'] = df_cross['CCF_classification'].fillna('X')
    df_cross['CORE_classification'] = df_cross['CORE_classification'].fillna('X')
    crosstab = pd.crosstab(df_cross.CCF_classification, df_cross.CORE_classification, \
            margins=True)[['A*', 'A', 'B', 'C', 'SP','X', 'All']]
    crosstab = crosstab.div(crosstab['All'],axis = 0)
    df_to_draw = crosstab[['A*', 'A', 'B', 'C']].iloc[[0,1,2]]
    df_to_draw.plot(kind='bar', stacked=True)
    return crosstab

def get_ccf_core_inner(df_ccf, df_dblp2ccf,df_core, df_dblp2core):
    df_ccf_tmp = pd.merge(df_ccf, df_dblp2ccf, left_on='CCF_id', right_on='ccf_CCF_id')
    df_core_tmp = pd.merge(df_core, df_dblp2core, left_on='CORE_id', right_on='core_CORE_id')
    df_ccf_core = pd.merge(df_ccf_tmp, df_core_tmp)
    df_ccf_core = df_ccf_core[['CCF_id',
       'CCF_type', 'CCF_classification',
       'computercategory_computerCategory_id', 'dblp2ccf_id', 'dblp_dblp_id', 'ccf_CCF_id',
       'CORE_id', 'CORE_type', 'CORE_classification', 'dblp2core_id', 'core_CORE_id']]
    df_ccf_core['CORE_classification'] = df_ccf_core['CORE_classification'].replace(
        ['Australasian', 'L', 'National', 'Not ranked', 'Unranked'], "SP")
    df_ccf_core['label'] = df_ccf_core['CCF_classification'] + ',' + df_ccf_core['CORE_classification']
    return df_ccf_core

def get_ccf_core_outer(df_ccf, df_dblp2ccf,df_core, df_dblp2core):
    df_ccf_tmp = pd.merge(df_ccf, df_dblp2ccf, left_on='CCF_id', right_on='ccf_CCF_id')
    df_core_tmp = pd.merge(df_core, df_dblp2core, left_on='CORE_id', right_on='core_CORE_id')

    df_ccf_core = pd.merge(df_ccf_tmp, df_core_tmp,how='outer')
    df_ccf_core = df_ccf_core[['CCF_id',
       'CCF_type', 'CCF_classification',
       'computercategory_computerCategory_id', 'dblp2ccf_id', 'dblp_dblp_id', 'ccf_CCF_id',
       'CORE_id', 'CORE_type', 'CORE_classification', 'dblp2core_id', 'core_CORE_id']]
    df_ccf_core['CORE_classification'] = df_ccf_core['CORE_classification'].replace(
        ['Australasian', 'L', 'National', 'Not ranked', 'Unranked'], "SP")
    df_ccf_core['CORE_classification'] = df_ccf_core['CORE_classification'].fillna('X')
    df_ccf_core['CCF_classification'] = df_ccf_core['CCF_classification'].fillna('X')
    df_ccf_core['label'] = df_ccf_core['CCF_classification'] + ',' + df_ccf_core['CORE_classification']
    return df_ccf_core


