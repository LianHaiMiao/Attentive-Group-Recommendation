# Attentive Group Recommendation

This is our implementation for the paper:

Da Cao, Xiangnan He, Lianhai Miao, Yahui An, Chao Yang, and Richang Hong. 2018. Attentive Group Recommendation.  In <em>The 41st International ACM SIGIR Conference on Research &#38; Development in Information Retrieval</em> (SIGIR '18). ACM, New York, NY, USA,  645-654.

In order to learn the group interest, we use attention mechanism to learn the aggregation strategy from data in a dynamic way.

**Please cite our SIGIR'18 paper if you use our codes. Thanks!** 

BibTeX:

```
@inproceedings{Cao2018Attentive,
 author = {Cao, Da and He, Xiangnan and Miao, Lianhai and An, Yahui and Yang, Chao and Hong, Richang},
 title = {Attentive Group Recommendation},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {645--654},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3209998},
 doi = {10.1145/3209978.3209998},
 acmid = {3209998},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {atention mechanism, cold-start problem, group recommendation, neural collaborative filtering, recommender systems},
}
```

## Environment Settings
We use the framework pytorch. 
- pytorch version:  '0.3.0'
- python version: '3.5'

## Example to run the codes.

Run AGREE:

```
python main.py
```

After training process, the value of HR and NDCG in the test dataset will be printed in command window after each optimization iteration.

Output:

```
AGREE at embedding size 32, run Iteration:30, NDCG and HR at 5
...
User Iteration 10 [449.8 s]: HR = 0.6216, NDCG = 0.4133, [1.0 s]
Group Iteration 10 [471.9 s]: HR = 0.5910, NDCG = 0.4005, [23.0 s]

```


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
