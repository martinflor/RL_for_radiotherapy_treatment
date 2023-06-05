# Reinforcement learning optimization for radiotherapy treatments
Reinforcement Learning Open-Source Project for Radiotherapy Treatments. 



## Key points

- [x] Reinforcement learning (RL) optimization
- [x] Pre-trained RL agents
- [x] Easy simulations
- [ ] Custom radiotherapy treatments 


## Description

We used a cellular model that results from the combination of a cellular model developed by [A. Jalalimanesh](https://www.sciencedirect.com/science/article/abs/pii/S0378475416300878) and another one developed by [O'Neil](https://scholarscompass.vcu.edu/etd/2831/). Reinforcement learning algorithms (Q-learning, Sarsa and Expected Sarsa) are used to provide optimizations on radiotherapy treatments. The goal of this work is to generate superior radiotherapy treatment plans concerning Tumor Control Probability (TCP), number of radiation fractions, total radiation dose, treatment duration, and healthy cell survival rate.

<details>
   <summary>Additional Informations</summary>
   <p>
This open-source project introduces an autonomous decision-making framework designed to evaluate whether adjustments are required in the ongoing radiotherapy treatment. Leveraging advanced machine learning algorithms, it analyzes tumor imaging during the treatment, fostering enhanced precision and effectiveness in radiotherapy procedures.</p>
</details>

## Usage

```anaconda
python application.py
```

<details>
   <summary>Title 1</summary>
   <p><p align="center">
<img src="app/images/EPL.jpg" width="100" height="100" border="10"/>
</p></p>
</details>

## Results
<h3 align="center">
|                | Radio of 0.8 | Radio of 0.7 | Radio of 0.6 | Cell cycle of 24h | Cell cycle of 20h | Cell cycle of 18h | Cell cycle of 16h |
|:--------------:|:------------:|:------------:|:------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|    Baseline    |    -52.4%    |    -51.0%    |    -33.2%    |       -55.0%      |       -62.3%      |       -59.0%      |       -50.3%      |
| Selected Agent |    +23.3%    |    -10.5%    |    -22.7%    |       -8.36%      |       +26.4%      |       +8.59%      |       +8.90%      |
</h3>

| Environment              | TCP (\%) | Dose (Gy) | Fractions (-) | Duration (h) | Survival (\%) |
|-----------------------------------|:-----------------:|:------------------:|:----------------------:|:---------------------:|:----------------------:|
| Radiosensitivity of 0.50 |        90.0       |    113.7 ± 21.77   |       34.7 ± 8.49      |     832.1 ± 203.76    |          93.5          |
| Radiosensitivity of 0.55 |       100.0       |    91.3 ± 16.92    |       27.6 ± 5.92      |     662.6 ± 142.06    |          93.7          |
| Radiosensitivity of 0.6  |       100.0       |    85.05 ± 19.30   |       26.2 ± 7.17      |     628.8 ± 172.1     |          93.8          |
| Radiosensitivity of 0.66 |       100.0       |    79.52 ± 14.80   |       24.9 ± 5.85      |     597.1 ± 140.4     |          94.6          |
| Radiosensitivity of 0.72 |       100.0       |    61.26 ± 9.71    |       18.96 ± 3.7      |      455.0 ± 88.2     |          93.9          |
| Radiosensitivity of 0.80 |       100.0       |    59.54 ± 11.22   |      18.25 ± 3.97      |     438.0 ± 95.30     |          93.7          |
| Cell Cycle of 8 hours    |        0.0        |    87.59 ± 33.4    |       35.21 ± 6.6      |     845.0 ± 159.0     |           0.0          |
| Cell Cycle of 10 hours   |        72.0       |    121.3 ± 36.2    |      40.52 ± 12.6      |     972.5 ± 303.4     |          62.0          |
| Cell Cycle of 12 hours   |        98.0       |    76.30 ± 18.68   |       24.37 ± 7.2      |     584.9 ± 172.7     |          84.7          |
| Cell Cycle of 14 hours   |       100.0       |    51.93 ± 11.73   |       16.48 ± 4.6      |     395.5 ± 109.5     |          87.3          |
| Cell Cycle of 16 hours   |       100.0       |     44.53 ± 7.8    |       14.25 ±3.0       |      342.0 ± 72.3     |          91.0          |
| Cell Cycle of 18 hours   |       100.0       |     47.0 ± 6.76    |       15.0 ± 2.65      |     360.2 ± 63.54     |          90.7          |
| Cell Cycle of 20 hours   |       100.0       |    47.9 ± 13.19    |       16.2 ± 7.48      |     387.8 ± 179.5     |          91.3          |
| Cell Cycle of 24 hours   |       100.0       |    42.6 ± 13.10    |       15.4 ± 8.55      |     369.6 ± 205.1     |          93.7          |

## License

[EPL]()
