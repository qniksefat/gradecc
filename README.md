# gradients-RL-task
Computing brain macroscale gradients in a motor Reinforcement Learning task.

# basics

## dataset

dataset is observed fMRI from 46 subjects during a motor Reinforcement Learning task. 
stored in `codes/RL_dataset_Mar2022/`

## atlas

stored in `data/Schaefer2018_1000Parcels_7Networks_order`

---

## data extraction
timeseries by Dan

## analysis
- correlation matrix by nilearn
- gradient analysis by brainspace
- seed connectivity analysis

## statistical analysis
- pairwise t-stats
- repeated measures ANOVA by pingouin
- false discovery rate correction

## plots
- connectivity matrix
- gradients
- statistics
- seed connectivity

---

## epic
the time periods during the task as follows.
data is stored in csv files.
each epic set to be `216` trials. tr ~ 2 seconds.
 
- rest: when subject is not doing the task.
297 trs. frist 3 trs dismissed.
- baseline: when subject is doing the task but no learning is in progress.
219 trs. frist 3 trs dismissed.
- learning: subject gets new feedback and starts learning the new criteria by herself.
619 trs. divied to early and late. this epic is trs 200:416.

    - early: when it seems that the subject doesn't know how the task has changed. first 3 trs dismissed => 3:219 trs.
    - late: when apparently most subject got it right. last 216 trs.

## indicator

maybe needs to be -> measure

shows the value of that region of interest (ROI). for example, gradient 1.

### gradient

transforming each ROI to another space by taking highest variance PCA. cite Margulies 2016.

### eccentricity

sum of top three gradient components squared. a distance to grad center.