# MPC Predictor SESS Benchmark
This repository contains the code, and evaluation scripts accompanying the paper “A Comprehensive Evaluation of Prediction Techniques and Their Influence on Model Predictive Control in Smart Energy Storage Systems”.

It provides a reproducible benchmark of forecasting models (Linear, XGBoost, RNN, TimeMixer, and TimesNet) for load, PV generation, and electricity price prediction. These models are evaluated both on standard error metrics (MSE, RMSE, MAE, MAPE, R²) and in combination with a model predictive controller (MPC) to assess their real-world impact on smart energy storage system (SESS) performance.

## Results

**Predictor Tuning**
* linear
    * 20250204_141500: Linear, load
    * 20250204_184612: Linear, PV
    * 20250204_222016: Linear, price
* xgboost
    * 20250128_182124: XGBoost, load
    * 20250129_211746: XGBoost, PV
    * 20250131_074032: XGBoost, price
* recurrent-net
    * 20250201_203130: RNN, load
    * 20250202_122543: RNN, PV
    * 20250203_061524: RNN, price
* time-mixer
    * 20250128_151138: TimeMixer, load
    * 20250129_071826: TimeMixer, PV
    * 20250131_143014: TimeMixer, price
* times-net
    * 20250128_135659: TimesNet, load
    * 20250131_104458: TimesNet, PV
    * 20250203_150716: TimesNet, price

**Predictor Test**
* linear
    * 20250207_072124: Linear, load
    * 20250207_072235: Linear, PV
    * 20250207_072340: Linear, price
* xgboost
    * 20250207_071942: XGBoost, load
    * 20250207_073746: XGBoost, PV
    * 20250207_072613: XGBoost, price
* recurrent-net
    * 20250207_070515: RNN, load
    * 20250207_070826: RNN, PV
    * 20250207_071556: RNN, price
* time-mixer
    * 20250206_205215: TimeMixer, load
    * 20250206_205225: TimeMixer, PV
    * 20250207_084358: TimeMixer, price
* times-net
    * 20250207_074319: TimesNet, load
    * 20250207_075541: TimesNet, PV
    * 20250211_072513: TimesNet, price

**Predictor MPC Evaluation**
* base: 20250208_092114
* perfect: 20250207_143021
* standard
    * linear: 20250722_203255
    * xgboost: 20250722_205950
    * recurrent-net: 20250722_204540
    * times-net: 20250723_032007
    * time-mixer: 20250723_030559
* retrain
    * linear: 20250723_033326
    * recurrent-net: 20250723_034609
    * time-mixer: 20250723_035851
    * times-net: 20250723_043739
    * xgboost: 20250725_124430

## Acknowledgments

Some of the prediction-model implementations are adapted from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Many thanks to that community for making their work available.

## Citation

**Pre-print:** <http://dx.doi.org/10.2139/ssrn.5158084>

If you use this repository or find it helpful in your own research, please cite:

```bibtex
@misc{Ludolfinger2025,
	address = {Rochester, NY},
	type = {{SSRN} {Scholarly} {Paper}},
	title = {A {Comprehensive} {Evaluation} of {Prediction} {Techniques} and {Their} {Influence} on {Model} {Predictive} {Control} in {Smart} {Energy} {Storage} {Systems}},
	doi = {10.2139/ssrn.5158084},
	language = {en},
	author = {Ludolfinger, Ulrich and Hamacher, Thomas and Martens, Maren},
	month = feb,
	year = {2025},
}
```

## Licensing

* The code and documentation are released under the MIT License.
* The electricity price data in `res/data/ee_prices.csv` are © Bundesnetzagentur | SMARD.de and redistributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
* The files `opsd_building_*.csv` in `res/data` originate from the [Open Power System Data](https://data.open-power-system-data.org/household_data/2020-04-15/) project and are also redistributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.