<script>
import sepsis from 'results/2025-11-22_00-33_Sepsis_Cases_batch/2025-11-22_00-33_Sepsis_Cases_summary_table.md'
import reference from './parts/reference.md'
</script>

# Synthetic Datasets

## Deterministic

### Alternating Bits
- Periodic sequence, Cycle: 111000

| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 2/3               |
| 2           | 2/3               |
| >2          | 1                 |

![alt text](results/final/2025-12-03_11-08_Synthetic111000_batch/2025-12-03_11-08_Synthetic111000_accuracy_by_window.png)




<!-- 1st ![alt text](image.png) -->
<!-- 2nd ![alt text](image-4.png) -->

- Periodic sequence, Cycle: 11100


| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 3/5 = 0.6         |
| 2           | 4/5 = 0.8         |
| >2          | 1                 |

<!-- ![alt text](image-1.png) -->
![alt text](results/final/2025-12-03_15-38_Synthetic11100_batch/2025-12-03_15-38_Synthetic11100_accuracy_by_window.png)

## Randomized
### Random Decision 

- xx{-x}: Alphabet={0,1}, Words={110, 001}. Name: Random_Decision_win2

| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 1/2 = 0.5         |
| 2           | 2/3 = 0.6667      |
| 3           | 2/3 = 0.6667      |
| >3          | 5/6 = 0.8333      |

<!-- ![alt text](image-2.png) -->
<!-- ![alt text](image-7.png) -->
![alt text](Random_Decision_win2.png)


<!-- - win 3: Alphabet={0,1,2}, Words={012, 021, 102, 120, 201, 210}

| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 4/9 = 0.4444   ???  |
| >1          | 11/18 = 0.6111 ???  |

<!-- ![alt text](image-3.png) -->
<!-- ![alt text](image-5.png) -->


- x1x0: Alphabet={0,1}, Words={1110, 0100}. Name: x1x0

| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 1/2 =   0.5        |
| 2           | 5/8 =  0.625       |
| 3           | 13/16 = 0.8125     |
| 4           | 13/16 = 0.8125     |
| >4          | 7/8 =   0.8750     |

![alt text](results/final/2025-12-03_15-51_x1x0_batch/2025-12-03_15-51_x1x0_accuracy_by_window.png)

- x10x01: Alphabet={0,1}, Words={110101, 010001}. Name: x10x01

| Window size | Theoretical bound |
|:-------------|:------------------|
| 1           | 2/3   = 0.6667     |
| 2           | 2/3   = 0.6667     |
| 3           | 5/6   = 0.8333     |
| 4           | 5/6   = 0.8333     |
| 5           | 7/8   = 0.8750     |
| 6           | 7/8   = 0.8750     |
| >6          | 11/12 = 0.9167     |

<!-- ![alt text](image-9.png) -->
![alt text](results/final/2025-12-04_00-57_x10x01_batch/2025-12-04_00-57_x10x01_accuracy_by_window.png)


# Real-World Datasets

## Sepsis Cases


> Typical window size = 4. [(Sepsis Cases - Results)](results/2025-12-04_12-08_Sepsis_Cases_batch/2025-12-04_12-08_Sepsis_Cases_summary_table.md)

| Model | Accuracy (%) |
|:-------------|:------------------|
| 5-gram           | 62.46 |
| soft-voting      | 65.68 |
| LSTM	           | 61.08 |
| LSTM_win5        | 64.90 |
| transformer	   | 65.77 |
| transformer_win2 | 59.69 |



![alt text](results/2025-12-04_12-08_Sepsis_Cases_batch/2025-12-04_12-08_Sepsis_Cases_accuracy_by_window.png)


## BPI Challenge 2012

> Typical window size = 6 [(BPI Challenge 2012 - Results)](results/2025-11-24_19-29_BPI_Challenge_2012_batch/2025-11-24_19-29_BPI_Challenge_2012_summary_table.md)

| Model | Accuracy (%) |
|:-------------|:------------------|
| 7-gram        | 85.73 |
| soft-voting   | 85.71 |
| LSTM	        | 86.04 |
| transformer	| 85.64 |

![alt text](results/2025-11-24_19-29_BPI_Challenge_2012_batch/2025-11-24_19-29_BPI_Challenge_2012_accuracy_by_window.png)



## BPI Challenge 2013

> Typical window size = 6 [(BPI Challenge 2013 - Results)](results/2025-11-25_17-33_BPI_Challenge_2013_batch/2025-11-25_17-33_BPI_Challenge_2013_summary_table.md)

| Model | Accuracy (%) |
|:-------------|:------------------|
| 7-gram        | 74.31 |
| soft-voting   | 73.83 |
| LSTM	        | 74.06 |
| transformer	| 73.57 |

![alt text](results/2025-11-25_17-33_BPI_Challenge_2013_batch/2025-11-25_17-33_BPI_Challenge_2013_accuracy_by_window.png)
