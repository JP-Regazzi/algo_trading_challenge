# Two-Way Trading Algorithm for Advanced Algorithms Competition at CentraleSupélec

This repository contains a two-way trading algorithm designed for the [Advanced Algorithms Competition on Codabench](https://www.codabench.org/competitions/2752/#/participate-tab), hosted by CentraleSupélec.

## Overview

The goal of this algorithm is to maximize profits through a strategic, competitive approach to two-way trading, specifically tailored to online trading constraints. The algorithm considers factors like price thresholds, competitive ratios, and optimal trading decisions to execute up to a defined limit of buy/sell transactions within a fluctuating price range.

### Problem Description
Given a stock that fluctuates within a price range, the algorithm aims to maximize the value of a limited initial investment by buying at low points and selling at high points across a set period. 

The core challenge is to make profitable trades using only the information available up to the current day (an online setting), where we do not have access to future price data.

### Key Features of the Algorithm

- **Threshold-based Trading**: The algorithm sets buy and sell thresholds, specifically \( M^{1/3} \) for buys and \( M^{2/3} \) for sells, where \( M \) is the maximum price in the range.
- **Stop-Loss Mechanism**: A stop-loss mechanism triggers sales to prevent losses if the price falls too low after reaching a peak.
- **Optimal Competitive Ratio**: This competitive, online algorithm approximates an optimal offline strategy, achieving a performance that minimizes the loss factor relative to the offline optimum.
  
### Parameters
- **Price Bounds (`m` and `M`)**: Known minimum and maximum stock prices.
- **`longueur`**: Total trading period length.
- **`max_trade_bound`**: The maximum allowable trades.
- **`taux_achat` and `taux`**: Current stock price and transaction state.
  
### Functions
- **`two_way_trading_online`**: The main function that processes daily trades, using observed minimum and maximum prices to set buy and sell thresholds, thereby maximizing the final capital.

## How to Use

1. Run the script, which imports input data for each test case from the specified `input_dir`.
2. The algorithm initializes with the initial capital and executes trades based on the price data over the specified period.
3. Results are saved to `output_dir`, detailing the score ratio for each test instance and the average score ratio across instances.

For more details on the competition, rules, and evaluation metrics, visit the competition page [here](https://www.codabench.org/competitions/2752/#/participate-tab).