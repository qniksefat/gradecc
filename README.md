# gradecc
Brain Large-scale manifolds (Gradients) in a Motor Reinforcement Learning task.

## Task

## Dataset

We had 46 subjects in functional MRI during a Motor Reinforcement Learning.
Stored as different files in `RL_dataset_Mar2022/`.

Seven subjects are remove due to behavioural issues. It's marked in `./data/participants.tsv`.
Also, subject SH1 excluded for not having subcortical data. Total of 8.

### Atlas
Cortical atlas is stored in `./data/Schaefer2018_1000Parcels_7Networks_order`.

### Time-series extraction
Cortical by Dan Gale. Subcortical/Cerebellum by Corson Areshenkoff.

### Epoch
Time-periods during the task as follows.
Each epoch is set to be **216 time-trials** of each ~ 2 seconds. 
Other time-trials dismissed.
 
- `rest` Subject is not doing the task.
297 trs. First 3 trs dismissed.
- `baseline` Subject is doing the task but no reward is given.
219 trs. First 3 trs dismissed.
- `learning` Subject starts getting rewards.
619 trs. Divided into early and late sections to differentiate learned period.
    - `early` When subject starts knowing how the task has changed. First 3 trs dismissed 
     => 3:219 trs.
    - `late` When some subjects got it right. The last 216 trs.


## Analysis
- Correlation matrix by Nilearn
- Gradient analysis by Brainspace
  - `measure` 
Any value for a brain region. 
For example, value for gradient 2 on for 7Networks_LH_Vis_3.
  - `eccentricity`
Euclidian distance to the center of PCA space. Sum of top 3 or 4 gradient components squared.

- Behavioural analysis.
 Based on task scores.


## Statistical analysis
- Pairwise t-tests
- Repeated-measures ANOVA by pingouin

  After including subcortical regions in gradient analysis, number of significant regions decreased from 57 to 50. No significant regions found in subcortex. 

- False discovery rate (FDR) correction
 by Benjamini-Hochberg method

## Post-hoc analysis
Seed connectivity of Regions of interest. 
Comparing shifts in functional connectivity pattern. 


## Plots
- Connectivity matrix
- Gradients
- Statistics
- Seed connectivity
- Behavioural


