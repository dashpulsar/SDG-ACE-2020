# SDG-ACE-2020
A Team project by Hayden B , Jason Verrall, Jaspal Panesar and Zhengyang Jin

Code link: [Here](code/)

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
We use Scrapy got some data from the SDG databases. We also created a journal discovery tool to search for suitable articles. This maintains the balance of the training data very well, which makes our model more robust.
So these data are labeled. We can use them for supervised learning. But this is not only our only method, we also think about directly using the provided data for unsupervised learning. Details will be introduced in "methods".

## 3. Methods  


### NLP Tools 2
"""
#### TF-IDF
Tf-idf by weighting the count of each word in a documentation with the inverse of its frequency across all documents.

We can weaken high-frequency words and strengthen low frequncy words, so that all words produce a quantitaive score.

#### Cosine Similarity
"""
---

### Why journal titles

* Previous approaches looked at classifying the entire corpus based on article content. 
* Value in allowing researchers to select potentially relevant journals for a field of interest
* Complements a whole-corpus approach as well as a quick knowledge discovery tool

---

### Possible uses
* Researcher who wants to identify potentially relevant journals for their field, e.g. to get ToCs as they are published for situational awareness
* Expand scope of journal content, e.g. going from region-specific publications to global or other regional publications
* Training expert systems 

---

### Worked example
Our user is an energy industry analyst/researcher

* Journal titles in dataset = **12,912**
* SDG7 keywords consolidated from Bergen, Elsevier, Sirius etc. = **511**
* Matched tokenised journal titles to keywords = **71**
* Top 5 most frequent words in 71 journals = **8,404**

---

List of 6 journal titles provided by user:
* 'Energy Policy', 'Energy Research & Social Science', 'Journal Of Cleaner Production', 'Nature Climate Change', 'Renewable Energy', 'Solar Energy'

* All 6 are in the list of 71 'possibly relevant' journals
* Possibly relevant journals = **65**

---

### Simple term frequency
Top words used in the 6 expert journals = 5,191

Use **cosine similarity** to match the group of 6 expert journals, to the most frequent terms in abstracts in the remaining 65 journals individually

---

![](https://i.imgur.com/TGo2qxs.png)

So results aren't great!

The mean H score for the top 5 journals here is **64.8** 

(https://www.scimagojr.com/journalrank.php)

---

Use cosine similarity to match as before, based on the **TF-IDF score** for each term

![](https://i.imgur.com/Te3AGp0.png)

This looks better even to a non-expert; mean H score for the top 5 journals is **88.2** 

---


#### Unsupervised learning
Other one is use expert knowledge and word embedding for unsupervised text classification.This method is inspired by an ACL 2019 paper, [《Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings》](https://www.aclweb.org/anthology/P19-1036/)
For details, please click: [Here (Unsupervised learning)](#jump2)

### <span id="jump2">3.2 Unsupervised learning</span>

Consider that in many cases, the cost of manually labeling data is often very large. We need some unsupervised or semi-supervised ways to solve the problem, which can save a lot of resources for other tasks.
The main ideas are:  

1. Clean the document d to generate a vector V(d*) representing the document;  

2. The text category L is cleaned, expanded, and filtered to generate the corresponding category lexicon, and the lexicon is used to generate the vector V(L*) representing the category L;  

3. Finally, similarity(V(d*),V(L*)), which category L has the highest similarity between the text, belongs to this category.  
![png](graph/unspvzd_model.png)



```python

import torch
import torch.nn as nn

class SDGs_unsuper_Model(nn.Module):
    def __init__(self, n_layer, emb_size, h_size, dr_rate):

        self.dr_rate = dr_rate
        self.Ws_share = nn.Linear(h_size, 1, bias=False)
        self.lstm1 = nn.LSTM(input_size=emb_size,hidden_size=h_size,num_layers=n_layer,batch_first=True,dropout=dr_rate)
        self.lstm2 = nn.LSTM(input_size=emb_size,hidden_size=h_size,num_layers=n_layer,batch_first=True,dropout=dr_rate)
        self.dropout = nn.Dropout(p=dr_rate)

      



```

#### 3.2.1 Data Cleaning
It is basically similar to the previous preprocessing method.

####  3.2.2 Enrichment
This step is carried out for label, and its main purpose is to expand the category thesaurus through four specific methods, specifically:  

1. Use experts or search engines to provide 3-5 representative words for each category;  

2. Use WordNet to add synonyms and synonyms corresponding to the word found in the previous step into the thesaurus;  

3. Use the existing category thesaurus to find the representative documents of each category (threshold 70%), and add the words in the document to the category thesaurus;  

4. Using Word Embedding, find some similar words to add to the thesaurus;  

PS: The words found in each step must have appeared in the document;  



#### 3.2.3 Consolidation
Consolidation refers to filtering out some non-obvious words in the category thesaurus found in the enrichment step, and leaving high-quality words. The filtering standard is judged by the following formula:

![png](graph/formula1.png)  

TF(w,c) is the frequency of word w in category c, the right side of the numerator is the average frequency of word w in all categories, and the denominator represents the variance of word w in the category other than c. When FAC(w,c) is lower than a certain threshold, the word w is deleted from the category.


#### 3.2.4 similarity
The last step is to calculate the cosine similarity between document d and category l. In vectorization, the LSA method is used to perform singular value decomposition using word-document and word-label matrices to generate their respective latent semantic spaces. Then use the respective generated vectors for cosine similarity calculation

## Result
Detiled result [ ](https://github.com/BlinkingStalker/SDG-ACE-2020/blob/master/result/classification_result.csv) 
```

```



## Reference
https://github.com/nestauk/im_tutorials/blob/master/notebooks/05_gtr_sdgs.ipynb

Amplayo, R. K. and Lapata, M. (2020) ‘Unsupervised Opinion Summarization with Noising and Denoising’, in. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 1934–1945. Available at: https://www.aclweb.org/anthology/2020.acl-main.175 (Accessed: 27 July 2020).

Fuso Nerini, F., Tomei, J., To, L.S. et al.(2018) Mapping synergies and trade-offs between energy and the Sustainable Development Goals.Nat Energy 3, 10–15.  

Gao, Y., Zhao, W. and Eger, S. (2020) ‘SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization’, in. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 1347–1354. Available at: https://www.aclweb.org/anthology/2020.acl-main.124 (Accessed: 27 July 2020).

Haj-Yahia, Z., Sieg, A. and Deleris, L. A. (2019) ‘Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings’, in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. ACL 2019, Florence, Italy: Association for Computational Linguistics, pp. 371–379. doi: 10.18653/v1/P19-1036.

LaFleur, M. (2019) ‘Art Is Long, Life Is Short: An SDG Classification System for DESA Publications’, SSRN Electronic Journal. doi: 10.2139/ssrn.3400135.

Simon, É., Guigue, V. and Piwowarski, B. (2019) ‘Unsupervised Information Extraction: Regularizing Discriminative Approaches with Relation Distribution Losses’, in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. ACL 2019, Florence, Italy: Association for Computational Linguistics, pp. 1378–1387. doi: 10.18653/v1/P19-1133.

Sovacool, Benjamin K., (2018), Success and failure in the political economy of solar electrification: Lessons from World Bank Solar Home System (SHS) projects in Sri Lanka and Indonesia, Energy Policy, 123, issue C, p. 482-493.  

Villavicencio Calzadilla, P., & Mauger, R. (2018). The UN's new sustainable development agenda and renewable energy: the challenge to reach SDG7 while achieving energy justice. Journal of Energy & Natural Resources Law, 36(2), 233-254.  

Wiseman, S. and Stratos, K. (2019) ‘Label-Agnostic Sequence Labeling by Copying Nearest Neighbors’, in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. ACL 2019, Florence, Italy: Association for Computational Linguistics, pp. 5363–5369. doi: 10.18653/v1/P19-1533.


