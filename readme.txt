# Hierarchical Dynamic Time Warping Methodology for Aggregating Multiple Geological Time Series

We extend Dynamic Time Warping (DTW), a widely used tool for measuring similarity between two separate input signals, to include a Hierarchical aggregation method (HDTW) for applying DTW on more than 2 input signals. 
Our approach to HDTW is not limited to aggregating up the hierarchies, but also indexes a unified “path matrix” for the original inputs, thus exposing similarity measures between the input signals and extrapolating the optimal match between them. 
As a use case for palaeo-reconstructions, we apply an HDTW-based peak finding algorithm on two published micro-scale measurements of speleothems from water limited environments

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The module code runs in a Python 3.6.6 environment developed using PyCharm
(Community edition v2020.1) and Spyder (v3.2.8). and the following dependencies: Numpy
(v1.15.4), Pandas (v0.24.1), Matplotlib (v3.0.2), Seaborn (v0.9.0). 

The underlying DTW module applied for HDTW is dtaidistance.dtw 1 by Wannes Meert, copyright 2017-2018 KU Leuven, DTAI Research Group under an Apache License version 2.0.

Please note that the HDTW code does not contain the underlying Distance Time Warping algorithm required for the HDTW mapping and aggregation processes. 
This algorithm needs to be passed and set to the module parameter ‘dtw_function_reference’, thus allows the researcher flexibility and full control over the underlying DTW function.

### Installing

Step 1 - Clone the HDTW.py file and placing it in the working folder.
Step 2 - Import the HDTW object from the file (e.g "from HDTW import HDTW")

Examples of using the file are included in HDTW_script.py

Example 1: 

From HDTW_script.py:
...
MNDS_HDTW = HDTW(MNDS1_HDTW_signals, dtw_function_reference=dta_dtw)
MNDS_HDTW.execute()
MNDS_HDTW.save('hdtw_mnds1.pkl')

MNDS_HDTW.HDTW_report_figure(save_filename='MNDS_HDTW_report.pdf',fig_size=(25, 31),
                           traverses_names=MNDS1_traverses_names)

sub_annual_age_model_figure(HDTW_object=MNDS_HDTW, 
                             age_estimation= 80,
                             age_estimation_deviation= 0,
                             age_scan_range= 0.1, 
                             threshold_value= 2, end_year=1991,
                             threshold_function=find_peaks, reverse_yesrs=True,
                             age_iterations= 100, figsize=(25, 31), 
                             adjust_best_fit_threshold=None,
                             save_name='MNDS_HDTW_sub_annual_age_model.pdf',
                             scale=50)

Further examples and instructions can be found in in supplement A of the electronic material of <paper ref> <DOI>

## Authors

Yuval Burstyn & Asaf Gazit

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Code reviewers

