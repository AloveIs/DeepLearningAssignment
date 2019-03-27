---
title: Assignment 1 - Report
author: Pietro Alovisi
---

# The Implementation

The implementation is done in Matlab, following the suggested project structure. 

I achieved good computational performance by using vectorized operations. In particular


# Results

Here are the results for the different configuration of paramers

## Experiment 1 

![This is the caption](../Result_Pics/b100e40eta1la0.pdf){ width=60% }

![This is the caption](../Result_Pics/b100e40eta1la0proto.pdf){ width=60% }


| $\lambda$ | eta | batches | epochs | Performance |
|:---------:|:---:|:-------:|:------:|:-----------:|
|     0     | 0.1 |   100   |   40   |$23.1\%  \pm 4\%$ |



23.1 acc
with high variance
(as low as 19 as high as 28)

## Experiment 2

![This is the caption](../Result_Pics/b100e40eta01la0.pdf){ width=60% }

![This is the caption](../Result_Pics/b100e40eta01la0proto.pdf){ width=60% }


| $\lambda$ | eta | batches | epochs | Performance |
|:---------:|:---:|:-------:|:------:|:-----------:|
|     0     | 0.01 |   100   |   40   |$34.5\%  \pm 2\%$ |


34,5 acc steady


## Experiment 3


![This is the caption](../Result_Pics/b100e40eta01la_1.pdf){ width=60% }

![This is the caption](../Result_Pics/b100e40eta01la_1proto.pdf){ width=60% }



| $\lambda$ | eta | batches | epochs |Performance |
|:---------:|:---:|:-------:|:------:|:-----------:|
|     0.1     | 0.01 |   100   |   40   |$34.3\%  \pm 0.6\%$ |



34,3 acc steady, std_dev less than 1%


## Experiment 4

![This is the caption](../Result_Pics/b100e40eta01la1_.pdf){ width=60% }

![This is the caption](../Result_Pics/b100e40eta01la1_proto.pdf){ width=60% }




| $\lambda$ | eta | batches | epochs |Performance |
|:---------:|:---:|:-------:|:------:|:-----------:|
|     1     | 0.01 |   100   |   40   |$21.5\%  \pm 0.5\%$ |



21,5 acc steady, std_dev less than 1%

