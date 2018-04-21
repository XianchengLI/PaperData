import pandas as pd
from pandas import DataFrame, Series
import os
import json
import codecs
import numpy as np

def get_df_paper(jsonfile):
	paper_set = []
	for paper in jsonfile:
		dic = {}
		dic['paper_id'] = paper['Id']
		dic['paper_title'] = paper['Ti']
		dic['publish_year'] = paper['Y']
		dic['citation_count'] = paper['CC']
		dic['conference_id'] = paper['C']['CId']
		dic['conference_name'] = paper['C']['CN']
		paper_set.append(dic)

	df_paper = DataFrame(paper_set)
	df_paper=df_paper[['paper_id', 'paper_title', 'publish_year', 'citation_count', 'conference_id', 'conference_name']]
	return df_paper

def get_df_a2p(jsonfile):
	author_set = []
	for paper in jsonfile:
		paper_id = paper['Id']
		AA = paper['AA']
		for author in AA:
			dic = {}
			dic['paper_id'] = paper_id
			dic['author_id'] = author['AuId']
			dic['author_name'] = author['AuN']
			try:
				dic['affiliation_id'] = author['AfId']
				dic['affiliation_name'] = author['AfN']
				dic['order'] = author['S']
			except:
				dic['affiliation_id'] = 0
				dic['affiliation_name'] = np.nan
				dic['order'] = author['S']
			author_set.append(dic)

	df_a2p = DataFrame(author_set)
	df_a2p = df_a2p[['paper_id', 'author_id', 'author_name', 'affiliation_id', 'affiliation_name', 'order']]
	return df_a2p

def get_df_fos(jsonfile):
	fos_set = []
	for paper in jsonfile:
		paper_id = paper['Id']
		try:
			F = paper['F']
			for fos in F:
				dic = {}
				dic['paper_id'] = paper_id
				dic['fos_id'] = fos['FId']
				dic['fos_name'] = fos['FN']

				fos_set.append(dic)
		except:
			continue

	df_fos = DataFrame(fos_set)
	df_fos = df_fos[['paper_id', 'fos_id', 'fos_name']]
	return df_fos

def get_df_relationship(jsonfile):
	ref_set = []
	for paper in jsonfile:
		paper_id = paper['Id']
		try:
			RId = paper['RId']
			for cur_id in RId:
				dic = {}
				dic['src_id'] = cur_id
				dic['dst_id'] = paper_id

				ref_set.append(dic)
		except:
			continue

	df_relationship = DataFrame(ref_set)
	df_relationship = df_relationship[['src_id', 'dst_id']]
	return df_relationship

#merge all json
def merge(filepath):
	pathDir =  os.listdir(filepath)
	tot_cnt = len(pathDir)
	alljson = []
	cnt = 0
	for allDir in pathDir:
		child = os.path.join('%s%s' % (filepath, allDir))
		f = open(child)
		d = json.load(f)
		f.close()
		dd = d['entities']
		alljson.extend(dd)
		cnt += 1
		print "-------------------------Finished: ", cnt*1.0/tot_cnt
	return alljson

if __name__ == '__main__':
	alljson = merge('./result/')
	df_paper = get_df_paper(alljson)
	df_a2p = get_df_a2p(alljson)
	df_fos = get_df_fos(alljson)
	df_relationship = get_df_relationship(alljson)
	df_paper.to_csv("./pandas_file/df_paper.csv", index=False, encoding='utf-8')
	df_a2p.to_csv("./pandas_file/df_a2p.csv", index=False, encoding='utf-8')
	df_fos.to_csv("./pandas_file/df_fos.csv", index=False, encoding='utf-8')
	df_relationship.to_csv("./pandas_file/df_relationship.csv", index=False, encoding='utf-8')
