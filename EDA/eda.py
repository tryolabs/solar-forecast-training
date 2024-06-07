# %%
import random

import cartopy.crs as crs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df_train = pd.read_parquet("../data/trainset_new.parquet")
df_test = pd.read_parquet("../data/testset_forecast.parquet")

df_loc_train = df_train[["latitude_rounded", "longitude_rounded"]].drop_duplicates()
print(
    f"train latitude range: {df_loc_train['latitude_rounded'].min()},"
    f" {df_loc_train['latitude_rounded'].max()}"
)
print(
    f"train longitude range: {df_loc_train['longitude_rounded'].min()},"
    f" {df_loc_train['longitude_rounded'].max()}"
)

df_loc_test = df_test[["latitude_rounded", "longitude_rounded"]].drop_duplicates()
print(
    f"test latitude range: {df_loc_test['latitude_rounded'].min()},"
    f" {df_loc_test['latitude_rounded'].max()}"
)
print(
    f"test longitude range: {df_loc_test['longitude_rounded'].min()},"
    f" {df_loc_test['longitude_rounded'].max()}"
)

start_date = df_train["date"].min()
end_date = df_train["date"].max()
# %%
df_train.columns
# %%
# plot random station
train_ids = df_train["ss_id"].unique()
test_ids = df_test["ss_id"].unique()
idx = random.choice(train_ids)
df_train[df_train["ss_id"] == idx][["average_power_kw", "date"]].plot(
    x="date", y="average_power_kw", title=f"{idx}"
)
# %%
# from visual inspection exclude the following panels
# 17693: inconsistency in time series
# 6611, 7750, 6927, 26865: only 0 entries
outliers_train = []
outliers_test = []
save = False
outliers_train.extend([17693, 6611, 7750, 6927, 26865])


# %%
class CleanOutliers:
    """calculate different statistics for the dataset and outliers to be removed"""

    def __init__(self, train: bool, outliers: list):
        """initialize arguments

        Args:
            train (bool): if True conditions for training set are used,
            else conditions for test set  are used
            outliers (list): list of outliers to start with
        """
        self.train = train
        self.outliers = outliers

    def apply_filter(self, df: pd.DataFrame, filter: dict) -> pd.DataFrame:
        """applies defined filter to the dataframe, keeps only the data defined in the filter

        Returns:
            pandas dataframe: dataframe after applying the filter
        """
        print(f"data length before filtering: {len(df)}")

        for key, value in filter.items():
            df = df[df[key] == value]
        self.ids = df["ss_id"].unique()
        self.df_station = pd.DataFrame(data={"ss_id": self.ids})
        print(f"remaining data length after filtering: {len(df)}")
        return df

    def get_night_energy(self, df: pd.DataFrame, criterion: dict) -> pd.DataFrame:
        """get different statistics for the energy at night

        Args:
            df (pd.DataFrame): dataframe containing the data
            criterion (dict): criterion to remove outliers

        Returns:
            pd.DataFrame: dataframe with statistics for each panel
        """
        df_night_sum = df.groupby("ss_id").apply(lambda x: self._night_sum(x)).reset_index()
        df_night_sum.rename(columns={0: "total_power_at_night_kw"}, inplace=True)
        df_night_mean = df.groupby("ss_id").apply(lambda x: self._night_mean(x)).reset_index()
        df_night_mean.rename(columns={0: "mean_power_at_night_kw"}, inplace=True)
        df_night_median = df.groupby("ss_id").apply(lambda x: self._night_median(x)).reset_index()
        df_night_median.rename(columns={0: "median_power_at_night_kw"}, inplace=True)
        self.df_station = df_night_sum.merge(df_night_mean, on="ss_id")
        self.df_station = self.df_station.merge(df_night_median, on="ss_id")

        idx_outliers = []
        for key, value in criterion.items():
            df_out = self.df_station[self.df_station[key] >= value]
            idx_outliers.extend(list(df_out["ss_id"].values))
            self.outliers.extend(idx_outliers)

        return self.df_station, idx_outliers

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """calculate some statistics for individual panels

        Args:
            df (pandas dataframe): complete dataframe with all panels

        Returns:
            pandas datafame: dataframe with statistics for each panel in ids
        """
        means = df.groupby("ss_id")["average_power_kw"].mean()
        means_winter = df[df["date"].dt.month.isin([12, 1, 2])]["average_power_kw"].mean()
        means_spring = df[df["date"].dt.month.isin([3, 4, 5])]["average_power_kw"].mean()
        means_summer = df[df["date"].dt.month.isin([6, 7, 8])]["average_power_kw"].mean()
        means_autumn = df[df["date"].dt.month.isin([9, 10, 11])]["average_power_kw"].mean()
        medians = df.groupby("ss_id")["average_power_kw"].median()
        stds = df.groupby("ss_id")["average_power_kw"].std()
        lats = df.groupby("ss_id")["latitude_rounded"].mean()
        lons = df.groupby("ss_id")["longitude_rounded"].mean()

        s_date = df.groupby("ss_id")["date"].min()
        e_date = df.groupby("ss_id")["date"].max()

        data = {
            "mean": means,
            "median": medians,
            "std": stds,
            "lat": lats,
            "lon": lons,
            "start_date": s_date,
            "end_date": e_date,
            "mean_winter": means_winter,
            "mean_spring": means_spring,
            "mean_summer": means_summer,
            "mean _autumn": means_autumn,
        }
        df_statistics = pd.DataFrame(data=data)
        self.df_station = df_statistics.merge(self.df_station, on="ss_id")
        return self.df_station

    def get_data_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        """get information about the data availability

        Args:
            df (pd.DataFrame): complete dataframe with all panels

        Returns:
            pd.DataFrame: dataframe with info for data availability for each panel in ids
        """
        data = {
            "ss_id": [],
            "total_length": [],
            "data_length": [],
            "time_span": [],
            "missing_values_total": [],
            "missing_values_span": [],
        }
        for id_ in self.ids:
            # missing values for entire time frame
            df_tmp = df[df["ss_id"] == id_]
            if df_tmp.empty:
                self.outliers.extend([id_])
            else:
                s_date = df_tmp["date"].min()
                e_date = df_tmp["date"].max()
                df_span = pd.DataFrame(
                    pd.date_range(start=s_date, end=e_date, freq="h"), columns=["date"]
                )
                # missing values over available time
                df_time = pd.DataFrame(
                    pd.date_range(start="2018-01-01", end="2021-11-09", freq="h"), columns=["date"]
                )
                # total nr of samples available
                da = len(df_time[df_time.date.isin(df_tmp.date)])
                data["ss_id"].append(id_)
                data["total_length"].append(len(df_time))
                data["data_length"].append(da)
                data["time_span"].append(len(df_span))
                data["missing_values_total"].append(len(df_time) - da)
                data["missing_values_span"].append(len(df_span) - da)
        df_data = pd.DataFrame(data=data)
        self.df_station = df_data.merge(self.df_station, on="ss_id")
        return self.df_station

    def remove_zero_sequences(self, df: pd.DataFrame, criterion: dict) -> pd.DataFrame:
        """remove sequences of zeros in target data

        Args:
            df (pd.DataFrame): complete dataframe with all panels

        Returns:
            pd.DataFrame: complete dataframe with all panels
            with samples removed that are a sequence of zeros
        """
        n = criterion["zero_sequence_length"]
        is_zero = df[target] == 0
        # Use cumsum to create groups of consecutive zeros
        zero_groups = is_zero.ne(is_zero.shift()).cumsum()

        # Filter groups where target is zero and count each group's size
        filtered_groups = df[is_zero].groupby(zero_groups).filter(lambda x: len(x) < n)
        valid_indices = df[is_zero].loc[filtered_groups.index].index

        # Combine indices of non-zero rows with valid zero sequence rows
        non_zero_indices = df[~is_zero].index
        all_valid_indices = non_zero_indices.union(valid_indices).sort_values()
        # Return the DataFrame with only the valid indices
        return df.loc[all_valid_indices]

    def remove_outliers(self, df_statistics: pd.DataFrame, criterion: dict) -> pd.DataFrame:
        """removes outliers depending on "criterion" from the training data

        Args:
            df_statistics (pd.DataFrame): dataframe with statistics for each panel in ids
            criterion (dict): criterion to remove outliers

        Returns:
            pd.DataFrame: dataframe with statistics for each panel in ids with outliers removed
            lsit: list of outliers ids
        """
        if self.train:
            # remove time series that have few data
            outlier_idx = df_statistics[df_statistics["time_span"] < criterion["min_time_span"]][
                "ss_id"
            ].values
            self.outliers.extend(outlier_idx)
            # remove data with more than p% missing values of available time span
            p = criterion["percentage_missing"]
            outlier_idx = df_statistics[
                df_statistics["missing_values_span"] / df_statistics["time_span"] > p
            ]["ss_id"].values
            self.outliers.extend(outlier_idx)
            # remove panels with median above p-percentile
            p = np.percentile(df_statistics["median"], q=[criterion["percentile"]])[0]
            outlier_idx = df_statistics[df_statistics["median"] > p]["ss_id"].values
            self.outliers.extend(outlier_idx)
            self.outliers = list(set(self.outliers))

            df_statistics = df_statistics[~df_statistics["ss_id"].isin(self.outliers)]
        return df_statistics, self.outliers

    def __call__(
        self, df: pd.DataFrame, night_criterion: dict, statistics_criterion: dict
    ) -> pd.DataFrame:
        """removes outliers from dataframe and returns statistics of the remaining panels

        Args:
            df (pd.DataFrame): complete dataframe with all panels

        Returns:
            pd.DataFrame: dataframe with statistics for each panel for all ids
            pd.DataFrame: dataframe with statistics for each panel in ids with outliers removed
            lsit: list of outliers ids
        """
        clean_outliers = CleanOutliers(train=self.train, outliers=self.outliers)
        df = clean_outliers.apply_filter(df, filter)
        df = clean_outliers.remove_zero_sequences(df, statistics_criterion)
        df_statistics, _ = clean_outliers.get_night_energy(df, night_criterion)
        df_statistics = clean_outliers.get_statistics(df)
        df_statistics = clean_outliers.get_data_availability(df)
        df_statistics_clean, outliers = clean_outliers.remove_outliers(
            df_statistics, statistics_criterion
        )
        return df_statistics, df_statistics_clean, outliers

    def _night_sum(self, df_loc):
        """get total sum of energy at night

        Returns:
            _type_: _description_
        """
        night_sum = df_loc[df_loc["is_day"] == 0][target].sum()
        return night_sum

    def _night_mean(self, df_loc):
        """get mean of energy at night

        Returns:
            _type_: _description_
        """
        night_mean = df_loc[df_loc["is_day"] == 0][target].mean()
        return night_mean

    def _night_median(self, df_loc):
        """get median of energy at night

        Returns:
            _type_: _description_
        """
        night_median = df_loc[df_loc["is_day"] == 0][target].median()
        return night_median


# %%
target = "average_power_kw"
# dictionaries to filter data
filter = {"is_day": 1}
night_criterion = {"total_power_at_night_kw": 100}
statistics_criterion = {
    "min_time_span": 720,
    "percentage_missing": 0.9,
    "percentile": 90,
    "zero_sequence_length": 30,
}

clean_outliers = CleanOutliers(train=True, outliers=outliers_train)
df_statistics_train, df_statistics_train_clean, outliers_train = clean_outliers(
    df_train, night_criterion, statistics_criterion
)
print(len(outliers_train))

clean_outliers = CleanOutliers(train=False, outliers=outliers_test)
df_statistics_test, df_statistics_test_clean, outliers_test = clean_outliers(
    df_test, night_criterion, statistics_criterion
)
print(len(outliers_test))
# %%
# boxplots
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
df_statistics_train.boxplot(column=["mean", "median", "std", "mean_summer"], ax=ax[0])
ax[0].set_title("train")
ax[0].set_ylim(0, 1)
df_statistics_train_clean.boxplot(column=["mean", "median", "std", "mean_summer"], ax=ax[1])
ax[1].set_title("train clean")
ax[1].set_ylim(0, 1)
ax[2] = df_statistics_test.boxplot(column=["mean", "median", "std", "mean_summer"], ax=ax[2])
ax[2].set_title("test")
ax[2].set_ylim(0, 1)
if save:
    plt.savefig("boxplot_only_day.png")
# %%
# spatial plots of statistical variables
var = "median"
df1 = df_statistics_train_clean
df2 = df_statistics_test
x1 = df1.lon
y1 = df1.lat
data1 = df1[var]
title1 = f"train - {var} ({len(df1)})"
x2 = df2.lon
y2 = df2.lat
data2 = df2[var]
title2 = f"test - {var} ({len(df2)})"
vmin = 0
vmax = np.max([data1.max(), data2.max()])

fig = plt.figure(figsize=(15, 6))
cm = plt.colormaps["RdYlBu"]
ax = fig.add_subplot(1, 2, 1, projection=crs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.left_labels = False
lat1, lon1, lat2, lon2 = 50.0, -5.7, 59.0, 1.8
ax.set_extent([lon1, lon2, lat1, lat2], crs=crs.PlateCarree())
sc = plt.scatter(
    x=x1, y=y1, c=data1, vmin=vmin, vmax=vmax, cmap=cm, s=2, alpha=0.8, transform=crs.PlateCarree()
)
plt.colorbar(sc)
plt.title(title1)
ax = fig.add_subplot(1, 2, 2, projection=crs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.left_labels = False
lat1, lon1, lat2, lon2 = 50.0, -5.7, 59.0, 1.8
ax.set_extent([lon1, lon2, lat1, lat2], crs=crs.PlateCarree())
sc = plt.scatter(
    x=x2, y=y2, c=data2, vmin=vmin, vmax=vmax, cmap=cm, s=2, alpha=0.8, transform=crs.PlateCarree()
)
plt.colorbar(sc)
plt.title(title2)
if save:
    plt.savefig(f"{var}_only_day.png")

# %%
