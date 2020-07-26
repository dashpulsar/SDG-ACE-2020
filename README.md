# SDG-ACE-2020
A Team project by Zhengyang Jin, Jason Verrall, Hayden B and Jaspal Panesar


## 1. Competition Introduction  
The Sustainable Development Goals (SDGs) are a collection of 17 global goals designed to be a "blueprint to achieve a better and more sustainable future for all".
The SDGs, set in 2015 by the United Nations General Assembly and intended to be achieved by the year 2030, are part of UN Resolution 70/1, the 2030 Agenda.

The Sustainable Development Goals are:  
1.No Poverty  
2.Zero Hunger  
3.Good Health and Well-being    
4.Quality Education    
5.Gender Equality  
6.Clean Water and Sanitation  
7.Affordable and Clean Energy  
8.Decent Work and Economic Growth  
9.Industry, Innovation, and Infrastructure  
10.Reducing Inequality  
11.Sustainable Cities and Communities  
12.Responsible Consumption and Production  
13.Climate Action  
14.Life Below Water  
15.Life On Land  
16.Peace, Justice, and Strong Institutions  
17.Partnerships for the Goals  
  
<br>
The goals are broad based and interdependent. The 17 sustainable development goals each has a list of targets which are measured with indicators.
In an effort to make the SDGs successful, data on the 17 goals has been made available in an easily-understood form. A variety of tools exist to track and visualize progress towards the goals.  


<br>
<br>
Our task is to use existing resources to create a project to contribute to the SDG program.  
Here, we use natural language processing technology to classify and cluster some data to reduce the workload of experts.

## 2. Resource 

### Official Resource
2018_WoS.csv  
This is a table contain publication code, journal name , topic , abstract of paper.

Columns:
0: Publication code. A unique code that identifies the publication.
1: Journal name.
2: Title.
3: Abstract.

### Extended Resources  
We use Scrapy got some data from the SDG databases.
These data are labeled. We can use them for supervised learning. But this is not only our only method, we also think about directly using the provided data for unsupervised learning. Details will be introduced in "methods".

## 3. Methods  
We mainly used two different methods.   
<br>
#### Supervised learning  
One is to use the labeled data of the extended data set for supervised learning.
We tried  KNeighborsClassifier and SVD techniques.For details, please click: [Here (Supervised learning)](#jump1)


#### Unsupervised learning
Other one is use expert knowledge and word embedding for unsupervised text classification.This method is inspired by an ACL 2019 paper, [《Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings》](https://www.aclweb.org/anthology/P19-1036/)
For details, please click: [Here (Unsupervised learning)](#jump2)

### <span id="jump1">3.1 Supervised learning</span>

#### 3.1.1 Preprocess
After testing, we found that in the data of 2018_WoS.csv, there are a lot of defective data. Among them, there are a total of 1,630,350 pieces of data, and 34,492 pieces of data lack abstract.
Therefore, we are faced with two choices, one is to divide the data into two data sets for separate classification, and the other is to fill in the default values. According to [LaFleur, M. (2019)](https://www.un.org/development/desa/publications/working-paper/wp159)
research, we decided to use the article title as an alternative to the default value. 
And we counted the number of journals, there are 12,912 journals in total.The number is too large and will seriously affect the classification accuracy. So we give up categorizing articles by journal name.  

#### 3.1.2 Data cleaning


### <span id="jump2">3.2 Unsupervised learning</span>



## Reference

