# CalCOFI_Oceanografic_data
Data Science with Julia

# 	Problem Statement
## Goal:
**Tackling climate crisis with Data Science: Water temperature prediction using 60 years of oceanographic data**

Increasing ocean temperatures severely affect marine species and ecosystems. Rising temperatures can contribute to coral bleaching and the loss of breeding grounds for marine fishes and mammals. Machine learning can be useful to predict what contributes to water temperature increase and to mitigate the rising temperatures in a timely fashion.
For this experiment, I used the California Cooperative Oceanic Fisheries Investigations (CalCOFI) dataset that comprises of 60 years of oceanographic data, hosted on Kaggle: https://www.kaggle.com/sohier/calcofi
The CalCOFI data set represents the longest (1949-present) and most complete (more than 50,000 sampling stations) time series of oceanographic and larval fish data in the world. It includes data on temperature, salinity, oxygen, phosphate, silicate, nitrate and nitrite, chlorophyll, transmissometer, PAR, C14 primary productivity, phytoplankton biodiversity, zooplankton biomass, and zooplankton biodiversity. We see if there is any relation between water salinity and water temperature. And I will focus on predicting water temperature by using a supervised machine learning model Linear Regression. 


### Resource available:

Resources available includes:

●	Over 850,000 tuple with 74 attributes records of Oceanographic data, will be structured into CSV  containing:

○	Temperature, salinity, oxygen, nutrients.

○	Water masses and currents

○	Primary production

○	Phyto- and Zooplankton biomass and biodiversity.

○	Meteorological observations.

○	Distribution and Abundance of Fish eggs and larvae.

○	Marine birds and mammal census; marine mammal acoustic recordings

○	Fisher’s acoustics.
   

●	Jupyter Notebook, which delivers a step by step output along with the documentation and visual insights.

●	Julia programming language ( 1.5.1 latest).

# Project Description


## Background:

The CalCOFI data set represents the longest (1949-present) and most complete (more than 50,000 sampling stations) time series of oceanographic data in the world. The physical, chemical and biological data collected at regular time and space intervals quickly became valuable for documenting climatic cycles in the California Current and a range of biological responses to them. CalCOFI research drew world attention to the biological response to the dramatic Pacific-warming event in 1957-58 and introduced the term “El Niño” into the scientific literature.

The data-rich environment provides an excellent opportunity to explore how oceanographic monitoring programs can support ecosystem management strategies that are adaptive to climate change. 

## Analysis needs to be done:

As in this model, we will see:
1.	Description: Primarily we focused on understanding oceanographic data and events that have happened in the past.
2.	Detection: Less focused on the past and more focused on the attributes that affect our target(Water Temperature). 
3.	Prediction: Focused on the future and predicting future behaviors Water Temperature.
4.	Optimization: Here we optimize our model by applying the Ordinary Linear Regression Model.
5.	Testing: Focused on testing behaviors of Water Temperature. 
4.1	Scope of the Work


This work answers the existing questions:

1.	Is there a relationship between water salinity & water temperature? 

2.	Can you predict the water temperature based on salinity?

3.	How reliable is our Machine Learning model?

Analysis of such huge data creates an enormous volume of heterogeneous data that can be valuable to a wide variety of oceanographic research and resource management applications. Recent interest in making the full range of data readily available online has led to an effort to understand holistically the current state of data management of CalCOFI data and lay a path towards building an integrated online access system. 

