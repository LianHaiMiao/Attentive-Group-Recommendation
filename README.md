# Attentive Group Recommendation

This is our implementation for the paper:

Da Cao, Xiangnan He, Lianhai Miao, Yahui An, Chao Yang & Richang Hong. 2018. Attentive Group Recommendation. In SIGIR

In order to learn the group interest, we use attention mechanism to learn the aggregation strategy from data in a dynamic way.

**Please cite our SIGIR'18 paper if you use our codes. Thanks!** 

we will provide the BibTeX soon.


## Environment Settings
We use the framework pytorch. 
- pytorch version:  '0.3.0'
- python version: '3.5'

## Run The Code

```
python main.py
```

## Parameter Tuning

we put all the papameters in the config.py



## Parameter Tuning

we put all the papameters in the config.py



## Dataset

We provide one processed dataset: CAMRa2011. 

Because we have another paper use the MaFengWo dataset are under reviewing, so we can't release MaFengWo dataset now.

group(user) train.rating:

* Train file.
* Each Line is a training instance: groupID(userID)\t itemID\t rating\t timestamp (if have)

test.rating:

* group(user) Test file (positive instances).
* Each Line is a testing instance: groupID(userID)\t itemID\t rating\t timestamp (if have)

test.negative

* group(user) Test file (negative instances).
* Each line corresponds to the line of test.rating, containing 100 negative samples.
* Each line is in the format: (groupID(userID),itemID)\t negativeItemID1\t negativeItemID2 ...
