This is the data and code used in paper:The Impact of Conference Ranking Systems in Computer Science: A Comparative Regression Analysis
MAg.py :  main part of code used in this paper, including calculation of several metrics and data preparation for regression analysis.
tools.py: functions to add some columns to the original data.
getters.py : functions to import data from Mysql database

csv files:
original data in folder OriginalRecords. See forms.txt for descriptions of it.
Author.rar, Conferences.rar, Paper.rar: data after pretreatment.

Our data is saved as .csv and can be read by using pandas.read_csv()
Our data only contains information relevant to our experiment, and we've deleted and added columns according to our needs.
Users of this data can check the original data for full information.

For data set Conferences, its data schema contains(other columns can be ignored):
'conference_id': conference ID in MAG
'conference_name': name of conference
'con_abbr': abbreviation of conference name
'CCF_classification': classification in CCF
'CCF_category':category in CCF
'CORE_classification': classification in CORE

For data set Paper, its data schema contains(other columns can be ignored):
'paper_id': paper ID in MAG
'publish_year': published year
'country': publishing country
'con_id': ID of conference where this paper was published
'citation_count': citation count received by this paper
'CCF_classification': the CCF classification of conference where this paper was published
'CCF_category': the CCF category of conference where this paper was published
'CORE_classification': the CORE	classification of conference where this paper was published

For data set Author, its data schema contains(other columns can be ignored):
'author_id': author ID in MAG
'affiliation_country' : the country of affiliation where the author works
'author_name': name of author
'index': index of author(calculation is based on citations in our data)

See OriginalData for full information and forms.txt for descriptions of it.

py files:
MAg.py contains functions used in the pretreatment. 
Tools.py contains functions used to add columns to original data.
Getters.py contains functions to retrieve data from Mysql database.
Get_paper.py, get_json.py, api.py contain functions used to retrieve data from microsoft academic search.
We use the API provided by MAS to download data.
Click https://academic.microsoft.com/#/faq for further information and explaination about this API.

It's suggested that users begin with the original data or use MAS API to get data which meet their needs.


