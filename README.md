# Fine-Grained Forecasting of COVID-19 Trends at the County Level in the United States

## Architecture
<p align="center">
    <img src=".\assets\Workflow.jpg" height="300" width="500">
</p>
A novel time-series deep learning-based approach to predict short-term infection trends related to COVID-19 in the US Counties. This state-of-the-art forecasting models consists of bidirectional LSTM deep learning structure and county clusters to adapt to sudden dynamic changes, efficiently improving the learning effectiveness of relevant features. This research offers a valuable short-term epidemic prediction framework to aid governments in formulating public health policies and curbing disease transmission.

## Installation
To install necessary library packages, run the following command in your terminal:
```
pip install -r requirements.txt
```

## Usage
* Clone the repo to your project folder by using the following commend:

    ``git clone https://github.com/kleelab-bch/FIGI-Net``


* Prepare the dataset as Excel file and copy to the ``Data`` folder. 
* Follow the order of codes (in the ``src`` folder)
  * Run ``1_Temporal_Clustering.py`` to obtain the cluster labels of US counties.
    * The Clustering labels will be saved to a new custom sheet in the original Excel file. 
  * Then run ``2_FIGINet_Prediction.py`` for model training and result forecasting.
    * If the user uses pretrained models , please set the parameter ``Use_Pretrained`` as True. 
* The forecasting results will be generated in ``Results`` folder 

## Note
- All the Covid-19 Confirmed Data of US Counties are from <a href="https://coronavirus.jhu.edu/">Center for Systems Science and Engineering (CSSE) at Johns Hopkins University</a>.
- The ``lib`` folder includes all dependencies required for the FIGInet workflow.
- All trained models are saved to the ``Model`` folder.
- The raw error data, consisting of the RMSE and RRMSE values of evaluated models, have been added to the ``Data`` folder for comparison and analysis.

## Citation
```
@article{Song2025FIGINet,
  author       = {Song, Tzu‑Hsi and Clemente, Leonardo and Pan, Xiang and Jang, Junbong and Santillana, Mauricio and Lee, Kwonmoo},
  title        = {Fine‑grained forecasting of COVID‑19 trends at the county level in the United States},
  journal      = {npj Digital Medicine},
  volume       = {8},
  number       = {1},
  pages        = {204},
  year         = {2025},
  doi          = {10.1038/s41746-025-01606-1},
  url          = {https://www.nature.com/articles/s41746-025-01606-1}
}
```
## Contact
If you have any question about the code or paper, please contact [Tzu-Hsi.Song@childrens.harvard.edu](mailto:Tzu-Hsi.Song@childrens.harvard.edu)
