# solar-training-prediction

This repository contains the tools needed to train a model for solar energy prediction. This work was a collaboration with [Open Collaboration Foundation (OCF)](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/tree/main).

The model was trained using the [ml-garden](https://github.com/tryolabs/ml-garden) library which was developed during this project.

# Installation

To train the model the ml-garden library needs to be installed using `poetry`. For instructions on how to install `poetry` can be, please refer to the [poetry documentation](https://python-poetry.org/docs/#installing-with-pipx). Additionally we need to install `XGBoost`, which is the model used for training.

Use the following lines of code to install the needed packages and set the environment.

```
poetry init

poetry add git+ssh://git@github.com:tryolabs/ml-garden.git

poetry add xgboost

poetry shell
```

# Data

The training data was downloaded from [open-meteo](https://open-meteo.com/). More specifically, hourly forecast data of the [historical weather API](https://open-meteo.com/en/docs/historical-weather-api) was used. The time period is restricted by the availabilty of the target solar enegery data of the panels and covers the time between 2018 and 2021. Additional information about the time, location and specifics about the panel are used. The weather features used are listed below, with the description given by open-meteo.

- Temperature at 2m (ºC): Air temperature at 2 meters above ground
- Relative Humidity at 2m (%): Relative humidity at 2 meters above ground
- Dewpoint at 2m (ºC): Dew point temperature at 2 meters above ground
- Precipitation (rain + snow) (mm): Total precipitation (rain, showers, snow) sum of the preceding hour
- Surface Pressure (hPa): Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology.
- Cloud Cover Total (%): Total cloud cover as an area fraction
- Cloud Cover Low (%): Low level clouds and fog up to 3 km altitude
- Cloud Cover Mid (%): Mid level clouds from 3 to 8 km altitude
- Cloud Cover High (%): High level clouds from 8 km altitude
- Wind Speed at 10m (km/h): Wind speed at 10, 80, 120 or 180 meters above ground. Wind speed on 10 meters is the standard level.
- Wind Direction (10m): Wind direction at 10 meters above ground
- Is day or Night: 1 if the current time step has daylight, 0 at night
- Direct Solar Radiation (W/m2): Direct solar radiation as average of the preceding hour on the horizontal plane and the normal plane (perpendicular to the sun)
- Diffusive Solar Radiation DHI (W/m2): Diffuse solar radiation as average of the preceding hour

The data was downloaded and transformed into a dataframe. The [training](https://drive.google.com/file/d/16b35aP2ML96-8B8CZ1KMjJxrUvAyS6LV/view?usp=sharing) and the [test](https://drive.google.com/file/d/1hYCsWnVWMsKujR-qBIjLlvW2rbPHeftE/view?usp=sharing) dataset are stored as `.parquet` files.

# Preprocessing

The panel data was carefully analyzed to detect outliers based on statistical analysis.

The exact preprocessing can be found in the [EDA](EDA/eda.py) folder.

Note, that no data was removed from the dataframe in this step. This analysis was only to identify data that should be removed. The actual removal is defined in the `config.json`.

# Train Model

The configuration file `config.json` provides the training parameters for the final model that is deployed. More details on how to setup a configuration file, please refer to the [documentation of ml-garden](https://github.com/tryolabs/ml-garden/blob/main/documentation/user_guide.md). To train the model with the provided `config.json`, run the script

```
python3 run_training.py
```

# Results

The pipeline execution returns a DataContainer object, in our script called `data`. This object contains the raw input data as a Pandas dataframe, which can be accessed using `data.raw`. This object contains the results of all steps performed in the pipeline the training. The prediction results can be accessed via `data.flow`.

The results are stored in the folder `runs`, which will be created during execution, if it doesn’t exist. The configuration file and a file containing the evaluation metrics are stored in this folder.
