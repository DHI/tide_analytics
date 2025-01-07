import numpy as np
from mikeio.eum import ItemInfo, EUMUnit, EUMType
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import textwrap
import mikeio
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize_scalar

from .helpers import (
    TideError,
    Tide,
    TidalErrors,
    TidalCharacteristics,
    Variable,
    NonAlternatingHWLWsError,
    NoSlackPointsFoundError,
    CurrentsToNoisyError,
    NotEnoughTidesError,
    NoHWLWsFoundError,
    NonMatchingSlackError,
    FallsWetError,
    NotEnoughWaterError,
    FallsPartiallyDryError,
    FallsPartiallyWetError,
)


class TidalSeries:
    def __init__(
        self,
        surface_elevation: mikeio.DataArray,
        current_speed: mikeio.DataArray | None = None,
        current_direction: mikeio.DataArray | None = None,
        velocity_x: mikeio.DataArray | None = None,
        velocity_y: mikeio.DataArray | None = None,
        SE_DIF: pd.Timedelta | None = pd.Timedelta("2h"),
        SE_PROM: float | None = None,
        CS_DIF: pd.Timedelta | None = None,
        CS_PROM: float | None = None,
        CD_DIF: pd.Timedelta | None = pd.Timedelta("1h"),
        MIN_TIDES: float = 3,
        MATCH_TOL: pd.Timedelta = None,
        THRSLD_WET_ERROR: float = 0.4,
        TOL_WET_ERROR: float = 1e-4,
        THRSLD_WET_WARNING: float = 0.05,
        THRSLD_NOWATER: float = 0.25,
        MIN_DATA: int = 10,
        DBSCAN_EPS: float = 10,
        DBSCAN_MINSAMPLES: float = 2,
        ignore_error_types: str | TideError | list[TideError] | None = None,
        raise_error_types: TideError | list[TideError] | None = None,
        error_support: bool = True,
    ):
        if ignore_error_types is not None and raise_error_types is not None:
            raise ValueError(
                "Either ignore_error_types or raise_error_types can be provided."
            )

        if isinstance(ignore_error_types, str):
            if ignore_error_types not in ["all_tidal", "all"]:
                raise ValueError(
                    "ignore_error_types must be one of ['all_tidal', 'all'] or any TideError."
                )

        self.SE_DIF = SE_DIF
        self.SE_PROM = SE_PROM
        self.CS_DIF = CS_DIF
        self.CS_PROM = CS_PROM
        self.CD_DIF = CD_DIF
        self.MIN_TIDES = MIN_TIDES
        self.MATCH_TOL = MATCH_TOL
        self.THRSLD_NOWATER = THRSLD_NOWATER
        self.THRSLD_WET_ERROR = THRSLD_WET_ERROR
        self.THRSLD_WET_WARNING = THRSLD_WET_WARNING
        self.TOL_WET_ERROR = TOL_WET_ERROR
        self.MIN_DATA = MIN_DATA
        self.DBSCAN_EPS = DBSCAN_EPS
        self.DBSCAN_MINSAMPLES = DBSCAN_MINSAMPLES

        self.ignore_error_types = ignore_error_types
        self.raise_error_types = raise_error_types
        self.error_support = error_support

        self.tidal_errors = TidalErrors()
        self.tidal_characteristics = TidalCharacteristics()

        self.surface_elevation, self.current_speed, self.current_direction = (
            self.parse_input(
                surface_elevation,
                current_speed,
                current_direction,
                velocity_x,
                velocity_y,
            )
        )

        self.surface_elevation, self.current_speed, self.current_direction = (
            self.exclude_warmup(
                self.surface_elevation, self.current_speed, self.current_direction
            )
        )
        self.errors_to_skip = self._get_errors()
        self.tidal_analysis()

    def tidal_analysis(self):
        """
        Performs tidal analysis on the surface elevation data.

        This method performs the following tasks:

        1. Checks if the if the element falls dry or wet.
        2. Calculates the high and low tide levels (HWLWs).
        3. Identifies the slack values using current speed and direction data.
        4. Matches the slack values to the HWLWs.
        5. Calculates the tidal characteristics, such as the tidal range, tidal period,
        and tidal amplitude.
        6. Checks for errors and warnings, and writes them to a dictionary.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If an error occurs during the analysis.
        """
        try:
            try:
                self._check_data()
                self.fallsdry = self._fallsdry()
                self.fallswet = self._fallswet()
                self.only_SE = False

                self.HWLWs = self.get_HWLWs()

                if self.fallsdry:
                    self.tides = self.get_tides(self.HWLWs, None)
                    self.aggregate_tidal_characteristics(self.tides)
                    raise FallsPartiallyDryError("The element falls partially dry.")
                if self.fallswet:
                    self.tides = self.get_tides(self.HWLWs, None)
                    self.aggregate_tidal_characteristics(self.tides)
                    raise FallsPartiallyWetError("The element falls partially wet.")

                self.slack_CS = self.get_slack_by_CS()
                self.slack_CD = self.get_slack_by_CD()
                self.slack = self.match_slack_to_HWLWs(
                    self.HWLWs, self.slack_CS, self.slack_CD
                )

                self.tides = self.get_tides(self.HWLWs, self.slack)
                self.tidal_characteristics = self.aggregate_tidal_characteristics(
                    self.tides
                )
            except self.errors_to_skip as e:
                all_tide_errors = {
                    name: cls
                    for name, cls in globals().items()
                    if isinstance(cls, type)
                    and issubclass(cls, TideError)
                    and cls is not TideError
                }
                if type(e) in all_tide_errors.values():
                    setattr(self.tidal_errors, type(e).__name__, True)
                else:
                    self.tidal_errors.UnknownError = True
        except Exception as e:
            if self.error_support:
                self.plot_all_timeseries(save=True)
            raise e

    def __str__(self):
        return (
            f"Tide Characteristics:\n"
            f"{textwrap.indent(str(self.tidal_characteristics), '    ')}"
        )

    def get_tides(self, HWLWs, slack):
        tides = self.split_tides(HWLWs, slack)

        if len(tides) < self.MIN_TIDES:
            raise NotEnoughTidesError(
                f"A minimum of {self.MIN_TIDES} tides is required. Only {len(tides)} were found."
            )

        return tides

    def aggregate_tidal_characteristics(self, tides: list):
        tides_df = pd.DataFrame([tide.to_dict() for tide in tides])

        tidal_characteristics = TidalCharacteristics(
            MHW=tides_df["high_tide"].mean(),
            MLW=tides_df["low_tide"].mean(),
            MTL=tides_df["mean_tide_level"].mean(),
            MAXECS=tides_df["max_ebb_current"].max(),
            MAXFCS=tides_df["max_flood_current"].max(),
            MEANECS=tides_df["mean_ebb_current"].mean(),
            MEANFCS=tides_df["mean_flood_current"].mean(),
            MTR=tides_df["tidal_range"].mean(),
            ECD=tides_df["ebb_current_duration"].mean(),
            FCD=tides_df["flood_current_duration"].mean(),
            ED=tides_df["ebb_duration"].mean(),
            FD=tides_df["flood_duration"].mean(),
        )

        return tidal_characteristics

    def _imbalance(self, h, surface_elevation):
        above = np.maximum(0, surface_elevation - h).sum()
        below = np.maximum(0, h - surface_elevation).sum()
        return abs(above - below)

    def split_tides(self, HWLWs: pd.DataFrame, slack: pd.DataFrame):
        tides = []

        se = self.surface_elevation.data.name

        if self.fallsdry or self.fallswet:
            for index, high_tide in HWLWs.iterrows():
                tides.append(
                    Tide(
                        high_tide=high_tide[se],
                        high_tide_time=index,
                    )
                )

        elif self.only_SE:
            for (
                (index_1_HWLWs, row_1_HWLWs),
                (index_2_HWLWs, row_2_HWLWs),
                (index_3_HWLWs, row_3_HWLWs),
            ) in zip(
                HWLWs.iloc[:-2:2].iterrows(),
                HWLWs.iloc[1:-1:2].iterrows(),
                HWLWs.iloc[2::2].iterrows(),
            ):
                se_tide = self.surface_elevation.data.loc[index_1_HWLWs:index_3_HWLWs]
                mean_tide_level = minimize_scalar(
                    self._imbalance, args=(se_tide.values,)
                ).x

                tides.append(
                    Tide(
                        ebb_time=[index_2_HWLWs, index_3_HWLWs],
                        flood_time=[index_1_HWLWs, index_2_HWLWs],
                        ebb_duration=index_3_HWLWs - index_2_HWLWs,
                        flood_duration=index_2_HWLWs - index_1_HWLWs,
                        tidal_range=abs(
                            row_2_HWLWs[se] - 0.5 * (row_3_HWLWs[se] + row_1_HWLWs[se])
                        ),
                        high_tide=row_2_HWLWs[se],
                        high_tide_time=index_2_HWLWs,
                        low_tide=row_1_HWLWs[se],
                        low_tide_time=index_1_HWLWs,
                        low_tide_2=row_3_HWLWs[se],
                        low_tide_2_time=index_3_HWLWs,
                        mean_tide_level=mean_tide_level,
                    )
                )

        else:
            for (
                (index_1_HWLWs, row_1_HWLWs),
                (index_2_HWLWs, row_2_HWLWs),
                (index_3_HWLWs, row_3_HWLWs),
                (index_1_slack, row_1_slack),
                (index_2_slack, row_2_slack),
                (index_3_slack, row_3_slack),
            ) in zip(
                HWLWs.iloc[:-2:2].iterrows(),
                HWLWs.iloc[1:-1:2].iterrows(),
                HWLWs.iloc[2::2].iterrows(),
                slack.iloc[:-2:2].iterrows(),
                slack.iloc[1:-1:2].iterrows(),
                slack.iloc[2::2].iterrows(),
            ):
                se_tide = self.surface_elevation.data.loc[index_1_HWLWs:index_3_HWLWs]
                mean_tide_level = minimize_scalar(
                    self._imbalance, args=(se_tide.values,)
                ).x

                tides.append(
                    Tide(
                        ebb_time=[index_2_HWLWs, index_3_HWLWs],
                        flood_time=[index_1_HWLWs, index_2_HWLWs],
                        ebb_duration=index_3_HWLWs - index_2_HWLWs,
                        flood_duration=index_2_HWLWs - index_1_HWLWs,
                        ebb_current_time=[index_2_slack, index_3_slack],
                        flood_current_time=[index_1_slack, index_2_slack],
                        ebb_current_duration=index_3_slack - index_2_slack,
                        flood_current_duration=index_2_slack - index_1_slack,
                        max_flood_current=float(
                            np.max(
                                self.current_speed.data.loc[index_1_slack:index_2_slack]
                            )
                        ),
                        max_ebb_current=float(
                            np.max(
                                self.current_speed.data.loc[index_2_slack:index_3_slack]
                            )
                        ),
                        mean_flood_current=float(
                            np.mean(
                                self.current_speed.data.loc[index_1_slack:index_2_slack]
                            )
                        ),
                        mean_ebb_current=float(
                            np.mean(
                                self.current_speed.data.loc[index_2_slack:index_3_slack]
                            )
                        ),
                        tidal_range=abs(
                            row_2_HWLWs[se] - 0.5 * (row_3_HWLWs[se] + row_1_HWLWs[se])
                        ),
                        high_tide=row_2_HWLWs[se],
                        high_tide_time=index_2_HWLWs,
                        low_tide=row_1_HWLWs[se],
                        low_tide_time=index_1_HWLWs,
                        low_tide_2=row_3_HWLWs[se],
                        low_tide_2_time=index_3_HWLWs,
                        mean_tide_level=mean_tide_level,
                    )
                )

        return tides

    def get_HWLWs(self) -> pd.DataFrame:
        """
        Calculates the high and low tide levels (HWLWs) from the surface elevation data.

        This method uses the `_find_peaks_troughs` method to identify the peaks and troughs
        in the surface elevation data, and then calculates the HWLWs based on these peaks
        and troughs.

        Args:
            None

        Returns:
            pd.DataFrame: A pandas DataFrame containing the HWLWs, including the surface
                elevation, current speed, and current direction data at each HWLW.

        Raises:
            NoHWLWsFoundError: If no HWLWs are found in the data.
            NonAlternatingHWLWsError: If the HWLWs are not alternating.
        """

        peaks, troughs = self._find_peaks_troughs(
            self.surface_elevation.data,
            time_difference=self.SE_DIF,
            prominence=self.SE_PROM,
        )

        tp = np.concatenate([peaks, troughs])

        HWLWs = pd.DataFrame(
            {
                self.current_speed.data.name: self.current_speed.data.iloc[tp].values,
                self.surface_elevation.data.name: self.surface_elevation.data.iloc[
                    tp
                ].values,
                self.current_direction.data.name: self.current_direction.data.iloc[
                    tp
                ].values,
            },
            index=self.current_speed.data.iloc[tp].index,
        )

        if HWLWs.empty:
            raise NoHWLWsFoundError(
                "No HWLWs have been found. Consider changing the SE_* parameters."
            )
        if len(HWLWs) < 3:
            raise NoHWLWsFoundError(
                f"Only {len(HWLWs)} HWLWs have been found. A Minimum of 3 is required. Consider changing the SE_* parameters."
            )

        HWLWs["type"] = np.concatenate([["HW"] * len(peaks), ["LW"] * len(troughs)])

        HWLWs = HWLWs.sort_index()

        if not self.fallsdry and not self.fallswet:
            if HWLWs["type"].iloc[0] != "LW":
                HWLWs = HWLWs.iloc[1:]
            if HWLWs["type"].iloc[-1] != "LW":
                HWLWs = HWLWs.iloc[:-1]

        alternates = all(
            HWLWs["type"].iloc[i] != HWLWs["type"].iloc[i + 1]
            for i in range(len(HWLWs) - 1)
        )

        if not self.fallsdry and not self.fallswet:
            if not alternates:
                self.HWLWs = HWLWs
                raise NonAlternatingHWLWsError(
                    "The types are not alternating in the HWLWs DataFrame. This is not supported."
                )
        else:
            HWLWs = HWLWs[HWLWs["type"] == "HW"]

        return HWLWs

    def get_slack_by_CS(self):
        """
        Calculates the slack times and types from the current speed.

        This method uses the `_find_peaks_troughs` method to identify the troughs in the current speed data,
        and calculates the slack type based on the surface elevation data.
        A DBScan is performed first to check whether the data is noisy or not.

        Args:
            None

        Returns:
            pd.DataFrame: A pandas DataFrame containing the slack times and
                types, or None if the element falls dry or wet.

        Raises:
            CurrentsToNoisyError: If the current direction data is too noisy to find
                the slack.
        """
        if not self.fallsdry and not self.fallswet:
            data = self.current_direction.data.values.reshape(-1, 1)
            dbscan = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MINSAMPLES)
            unique_clusters = np.unique(dbscan.fit_predict(data))

            if len(unique_clusters) < 2:
                self.only_SE = True
                self.tides = self.get_tides(self.HWLWs, None)
                self.tidal_characteristics = self.aggregate_tidal_characteristics(
                    self.tides
                )
                raise CurrentsToNoisyError(
                    f"The current direction data is to noisy to find the slack. Found {unique_clusters} clusters. {2} clusters are allowed"
                )

            _, troughs = self._find_peaks_troughs(
                self.current_speed.data,
                time_difference=self.CS_DIF,
                prominence=self.CS_PROM,
            )

            slack = pd.DataFrame(
                {
                    self.current_speed.data.name: self.current_speed.data.iloc[
                        troughs
                    ].values,
                    self.surface_elevation.data.name: self.surface_elevation.data.iloc[
                        troughs
                    ].values,
                    self.current_direction.data.name: self.current_direction.data.iloc[
                        troughs
                    ].values,
                },
                index=self.current_speed.data.iloc[troughs].index,
            )

            mean_se = np.mean(self.surface_elevation.data)
            slack["type"] = np.where(
                slack[self.surface_elevation.data.name] < mean_se, "to_flood", "to_ebb"
            )

            slack = slack.sort_index()

            return slack

    def get_slack_by_CD(self) -> pd.DataFrame:
        """
        Calculates the slack times and types from the current direction.

        This method uses the mean of the current direction data as a threshold to identify the transitions from below to above and vice versa,
        and calculates the slack type based on the surface elevation data.

        Args:
            None

        Returns:
            pd.DataFrame or None: A pandas DataFrame containing the slack times and
                types, or None if the element falls dry or wet.

        """
        if not self.fallsdry and not self.fallswet:
            mean_CD = self.current_direction.data.mean()

            below_to_above = (self.current_direction.data > mean_CD) & (
                self.current_direction.data.shift() <= mean_CD
            )
            above_to_below = (self.current_direction.data < mean_CD) & (
                self.current_direction.data.shift() >= mean_CD
            )

            transitions = below_to_above | above_to_below

            idcs = self.current_direction.data.index[transitions].unique().tolist()

            slack = pd.DataFrame(
                {
                    self.current_speed.data.name: self.current_speed.data.loc[
                        idcs
                    ].values,
                    self.surface_elevation.data.name: self.surface_elevation.data.loc[
                        idcs
                    ].values,
                    self.current_direction.data.name: self.current_direction.data.loc[
                        idcs
                    ].values,
                },
                index=self.current_speed.data.loc[idcs].index,
            )

            mean_se = np.mean(self.surface_elevation.data)
            slack["type"] = np.where(
                slack[self.surface_elevation.data.name] < mean_se, "to_flood", "to_ebb"
            )

            slack = slack.sort_index()

            return slack

    # def get_slack_by_CD(self):
    #     """
    #     Calculates the slack times and types from the current direction.

    #     This function checks for periods where the current direction is above or below
    #     the mean current direction and calculates the respective durations. The slack
    #     points are considered transitions from below to above or above to below the mean
    #     current direction, and only transitions that meet a minimum duration threshold,
    #     defined by `CD_DIF`, are considered valid. Based on the surface elevation at each
    #     transition, the function classifies the transitions as either "to_flood" or "to_ebb".

    #     TODO:
    #     - Modify the logic such that in the last transition will be considered correctly.
    #       Currently it skips the last transition if the following duration is smaller that CD_DIF.
    #       This is because the algorithm consideres the first value of a "duration" as a transition.

    #     Replaces the old algorithms that only checks for transitions but did not allow for check for their durations.
    #     Selecting CD_DIF to be pd.Timedelta("0min") (or something very small) replicates the behavior of the old algorihtm.

    #     Args:
    #         None

    #     Returns:
    #         pd.DataFrame or None: A pandas DataFrame containing the slack times and
    #             types, or None if the element falls dry or wet.

    #     """
    #     if not self.fallsdry and not self.fallswet:

    #         mean_CD = self.current_direction.data.mean()

    #         above = self.current_direction.data > mean_CD
    #         below = self.current_direction.data < mean_CD

    #         groups = (above != above.shift()).cumsum()
    #         durations = above.groupby(groups).transform('size') * (above.index[1] - above.index[0])
    #         valid = above & (durations >= self.CD_DIF)
    #         below_to_above = valid.loc[valid - valid.shift() == 1].index

    #         groups = (below != below.shift()).cumsum()
    #         durations = below.groupby(groups).transform('size') * (below.index[1] - below.index[0])
    #         valid = below & (durations >= self.CD_DIF)
    #         above_to_below = valid.loc[valid - valid.shift() == 1].index

    #         transitions = pd.concat([above_to_below.to_frame(), below_to_above.to_frame()]).sort_index().index.unique()

    #         slack = pd.DataFrame({
    #             self.current_speed.data.name: self.current_speed.data.loc[transitions].values,
    #             self.surface_elevation.data.name: self.surface_elevation.data.loc[transitions].values,
    #             self.current_direction.data.name: self.current_direction.data.loc[transitions].values
    #             }, index=self.current_speed.data.loc[transitions].index)

    #         mean_se = np.mean(self.surface_elevation.data)
    #         slack["type"] = np.where(
    #         slack[self.surface_elevation.data.name] < mean_se,
    #         "to_flood",
    #         "to_ebb")

    #         slack = slack.sort_index()

    #     return slack

    def match_slack_to_HWLWs(
        self, HWLWs: pd.DataFrame, slack_CS: pd.DataFrame, slack_CD: pd.DataFrame
    ):
        """
        Matches slack points to HWLWs based on the order of the indices.

        The slack points are matched to the HWLWs based on the order of the indices.

        Args:
            HWLWs (pd.DataFrame): HWLWs DataFrame containing the high and low tide levels and times.
            slack_CS (pd.DataFrame): Slack points DataFrame containing the current speed-based slack points.
            slack_CD (pd.DataFrame): Slack points DataFrame containing the current direction-based slack points.

        Returns:
            pd.DataFrame or None: A pandas DataFrame containing the matched slack points, or None if the element falls dry or wet.

        Raises:
            NoslackPointsFoundError: If no slack points were found.
        """
        if not self.fallsdry and not self.fallswet:
            HWLWs = HWLWs.sort_index()

            if slack_CD.empty and slack_CS.empty:
                raise NoSlackPointsFoundError(
                    "No slack points have been found. Consider changing Some parameters."
                )

            slack_CS = slack_CS.sort_index()
            slack_CD = slack_CD.sort_index()

            assert (
                not HWLWs.isna().any().any()
            ), "HWLWs contains NaN values. This shouldn't be possible."
            assert (
                not slack_CS.isna().any().any()
            ), "CS slack contains NaN values. This shouldn't be possible."
            assert (
                not slack_CD.isna().any().any()
            ), "CD slack contains NaN values. This shouldn't be possible."

            matched_slack_CS, sanity_CS = self._match_slack(slack_CS, HWLWs)
            matched_slack_CD, sanity_CD = self._match_slack(slack_CD, HWLWs)

            matched_slack = matched_slack_CS
            if sanity_CS:
                return matched_slack
            else:
                matched_slack = matched_slack_CD
                if sanity_CD:
                    return matched_slack
                else:
                    self.only_SE = True
                    self.tides = self.get_tides(self.HWLWs, None)
                    self.tidal_characteristics = self.aggregate_tidal_characteristics(
                        self.tides
                    )
                    self.slack = matched_slack_CD
                    raise NonMatchingSlackError(
                        f"slack points could not be matched to HWLWs. \nlength of slack: {len(matched_slack)}\nlength of HWLWs: {len(HWLWs)}\nSanity CS: {sanity_CS}\nSanity CD: {sanity_CD}"
                    )

    def _match_slack(self, slack: pd.DataFrame, HWLWs: pd.DataFrame):
        """
        Matches slack points to HWLWs based on the order of the indices.

        Within a certain tolerance, this method first finds the nearest index in slack for each index in HWLWs,
        and then checks if the resulting matched slack points have alternating types and the same length as HWLWs.

        Args:
            slack (pd.DataFrame): slack points DataFrame containing the current speed-based slack points.
            HWLWs (pd.DataFrame): HWLWs DataFrame containing the high and low tide levels and times.

        Returns:
            pd.DataFrame or None: A pandas DataFrame containing the matched slack points, or None if the element falls dry or wet.
            bool: A boolean indicating whether the matched slack points are sane, i.e. they have alternating types and the same length as HWLWs.
        """
        index = slack.index

        idcs = index.get_indexer(
            HWLWs.index.tolist(), method="nearest", tolerance=self.MATCH_TOL
        )
        idcs = idcs[idcs != -1]

        matched_slack = slack.iloc[idcs].sort_index()

        alternates = all(
            matched_slack["type"].iloc[i] != matched_slack["type"].iloc[i + 1]
            for i in range(len(matched_slack) - 1)
        )
        same_length = len(matched_slack) == len(HWLWs)

        sane = alternates and same_length

        return matched_slack, sane

    def exclude_warmup(
        self,
        surface_elevation: Variable,
        current_speed: Variable,
        current_direction: Variable,
    ) -> tuple[Variable, Variable, Variable]:
        """
        Exclude the warmup period from the given data.

        This method checks if a warmup period has been specified, and if so, it excludes the warmup period from the given data by slicing the DataFrames.

        Args:
            surface_elevation: The surface elevation data.
            current_speed: The current speed data.
            current_direction: The current direction data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The DataFrames with the warmup period excluded.
        """
        warmup = self._find_warmup(surface_elevation)

        if warmup is not None:
            surface_elevation.data = surface_elevation.data.loc[warmup:]
            current_speed.data = current_speed.data.loc[warmup:]
            current_direction.data = current_direction.data.loc[warmup:]

        return surface_elevation, current_speed, current_direction

    def plot_all_timeseries(self, figsize=None, save=False):
        if figsize is None:
            figsize = (10, 8)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

        ax1.plot(self.surface_elevation.data.index, self.surface_elevation.data.values)
        self.plot_surface_elevation(ax=ax1)
        # ax1.set_ylabel(self._label_txt(data = self.surface_elevation))
        ax1.set_ylabel("Surface Elevation [m]")

        ax2.plot(self.current_speed.data.index, self.current_speed.data.values)
        self.plot_current_speed(ax=ax2)
        # ax2.set_ylabel(self._label_txt(data = self.current_speed))
        ax2.set_ylabel("Current Speed [m/s]")

        ax3.plot(self.current_direction.data.index, self.current_direction.data.values)
        self.plot_current_direction(ax=ax3)
        # ax3.set_ylabel(self._label_txt(data = self.current_direction))
        ax3.set_ylabel("Current Direction [°]")

        fig.autofmt_xdate()
        if save:
            plt.tight_layout()
            fig.savefig("timeseries.png", dpi=300)

    def plot_surface_elevation(
        self,
        ax=None,
        plot_HWLWs: bool = True,
        plot_slack: bool = False,
        plot_tidal_range: bool = True,
        plot_tide_phase: bool = True,
        plot_mean_tide_level: bool = True,
        figsize=None,
    ) -> None:
        if ax is None:
            fig, ax = self._plot(self.surface_elevation, figsize=figsize)
        legend_handles: dict = {}
        y_min = ax.get_ylim()[0]

        if plot_tide_phase:
            if hasattr(self, "tides"):
                for tide in self.tides:
                    if tide.ebb_time is not None:
                        label = "Te"
                        label = "ED"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.ebb_time[0], tide.ebb_time[1], alpha=0.07, color='green', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_time[0],
                                xmax=tide.ebb_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.ebb_time[0], tide.ebb_time[1], alpha=0.07, color='green')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_time[0],
                                xmax=tide.ebb_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                    if tide.flood_time is not None:
                        label = "Tf"
                        label = "FD"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.flood_time[0], tide.flood_time[1], alpha=0.07, color='red', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.flood_time[0],
                                xmax=tide.flood_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.flood_time[0], tide.flood_time[1], alpha=0.07, color='red')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.flood_time[0],
                                xmax=tide.flood_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )

        if plot_tidal_range:
            if hasattr(self, "tides"):
                label = "TR"
                for tide in self.tides:
                    if tide.tidal_range is not None:
                        if label not in legend_handles:
                            legend_handles[label] = ax.vlines(
                                x=tide.high_tide_time,
                                ymin=tide.high_tide - tide.tidal_range,
                                ymax=tide.high_tide,
                                linewidth=0.7,
                                color="grey",
                                label=label,
                            )
                        else:
                            ax.vlines(
                                x=tide.high_tide_time,
                                ymin=tide.high_tide - tide.tidal_range,
                                ymax=tide.high_tide,
                                linewidth=0.7,
                                color="grey",
                            )

        if plot_mean_tide_level:
            if hasattr(self, "tides"):
                label = "Tmw"
                label = "MTL"
                for tide in self.tides:
                    if tide.mean_tide_level is not None:
                        if label not in legend_handles:
                            legend_handles[label] = ax.hlines(
                                y=tide.mean_tide_level,
                                xmin=tide.low_tide_time,
                                xmax=tide.low_tide_2_time,
                                linewidth=0.7,
                                color="grey",
                                linestyle="dashed",
                                label=label,
                            )
                        else:
                            ax.hlines(
                                y=tide.mean_tide_level,
                                xmin=tide.low_tide_time,
                                xmax=tide.low_tide_2_time,
                                linewidth=0.7,
                                color="grey",
                                linestyle="dashed",
                            )

        if plot_HWLWs:
            if hasattr(self, "HWLWs"):
                for index, HWLW in self.HWLWs.iterrows():
                    if HWLW["type"] == "HW":
                        label = "Thw"
                        label = "HW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.surface_elevation.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.surface_elevation.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if HWLW["type"] == "LW":
                        label = "Tnw"
                        label = "LW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.surface_elevation.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.surface_elevation.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                            )

        if plot_slack:
            if hasattr(self, "slack"):
                for index, slack in self.slack.iterrows():
                    if slack["type"] == "to_flood":
                        label = "Ke"
                        label = "Se"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.surface_elevation.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.surface_elevation.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if slack["type"] == "to_ebb":
                        label = "Kf"
                        label = "Sf"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.surface_elevation.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.surface_elevation.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1))

        if ax is None:
            plt.show()

    def plot_current_direction(
        self,
        ax=None,
        plot_slack: bool = True,
        plot_HWLWs: bool = False,
        plot_tide_current_phase: bool = True,
        plot_mean_current_direction: bool = True,
        figsize=None,
    ) -> None:
        if ax is None:
            fig, ax = self._plot(self.current_direction, figsize=figsize)
        legend_handles: dict = {}
        y_min = ax.get_ylim()[0]

        if plot_mean_current_direction:
            label = "MeanCd"
            if label not in legend_handles:
                legend_handles[label] = ax.axhline(
                    y=np.mean(self.current_direction.data),
                    color="grey",
                    linestyle="dashed",
                    alpha=0.5,
                    linewidth=0.7,
                    label=label,
                )
            else:
                ax.axhline(
                    y=np.mean(self.current_direction.data),
                    color="grey",
                    linestyle="dashed",
                    alpha=0.5,
                    linewidth=0.7,
                )

        if plot_tide_current_phase:
            if hasattr(self, "tides"):
                for tide in self.tides:
                    if tide.ebb_current_time is not None:
                        label = "Tce"
                        label = "ECD"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.ebb_current_time[0], tide.ebb_current_time[1], alpha=0.07, color='green', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_current_time[0],
                                xmax=tide.ebb_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.ebb_current_time[0], tide.ebb_current_time[1], alpha=0.07, color='green')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_current_time[0],
                                xmax=tide.ebb_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                    if tide.flood_current_time is not None:
                        label = "Tcf"
                        label = "FCD"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.flood_current_time[0], tide.flood_current_time[1], alpha=0.07, color='red', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.flood_current_time[0],
                                xmax=tide.flood_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.flood_current_time[0], tide.flood_current_time[1], alpha=0.07, color='red')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.flood_current_time[0],
                                xmax=tide.flood_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )

        if plot_slack:
            if hasattr(self, "slack"):
                for index, slack in self.slack.iterrows():
                    if slack["type"] == "to_flood":
                        label = "Ke"
                        label = "Se"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.current_direction.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.current_direction.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if slack["type"] == "to_ebb":
                        label = "Kf"
                        label = "Sf"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.current_direction.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.current_direction.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                            )

        if plot_HWLWs:
            if hasattr(self, "HWLWs"):
                for index, HWLW in self.HWLWs.iterrows():
                    if HWLW["type"] == "HW":
                        label = "Thw"
                        label = "HW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.current_direction.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.current_direction.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if HWLW["type"] == "LW":
                        label = "Tnw"
                        label = "LW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.current_direction.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.current_direction.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        if ax is None:
            plt.show()

    def plot_current_speed(
        self,
        ax=None,
        plot_slack: bool = True,
        plot_HWLWs: bool = False,
        plot_current_characteristics: bool = True,
        plot_tide_current_phase: bool = True,
        figsize=None,
    ) -> None:
        if ax is None:
            fig, ax = self._plot(self.current_speed, figsize=figsize)
        legend_handles: dict = {}
        y_min = ax.get_ylim()[0]

        if plot_tide_current_phase:
            if hasattr(self, "tides"):
                for tide in self.tides:
                    if tide.ebb_current_time is not None:
                        # label = "Tce"
                        label = "ECD"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.ebb_current_time[0], tide.ebb_current_time[1], alpha=0.07, color='green', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_current_time[0],
                                xmax=tide.ebb_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.ebb_current_time[0], tide.ebb_current_time[1], alpha=0.07, color='green')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.ebb_current_time[0],
                                xmax=tide.ebb_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="green",
                                label=label,
                            )
                    if tide.flood_current_time is not None:
                        # label = "Tcf"
                        label = "FCD"
                        if label not in legend_handles:
                            # legend_handles[label] = ax.axvspan(tide.flood_current_time[0], tide.flood_current_time[1], alpha=0.07, color='red', label=label)
                            legend_handles[label] = ax.hlines(
                                y=y_min,
                                xmin=tide.flood_current_time[0],
                                xmax=tide.flood_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )
                        else:
                            # ax.axvspan(tide.flood_current_time[0], tide.flood_current_time[1], alpha=0.07, color='red')
                            ax.hlines(
                                y=y_min,
                                xmin=tide.flood_current_time[0],
                                xmax=tide.flood_current_time[1],
                                linewidth=5,
                                alpha=0.4,
                                color="red",
                                label=label,
                            )

        if plot_current_characteristics:
            if hasattr(self, "tides"):
                for tide in self.tides:
                    if tide.max_flood_current is not None:
                        label = "MaxFCS"
                        if label not in legend_handles:
                            legend_handles[label] = ax.hlines(
                                y=tide.max_flood_current,
                                xmin=tide.low_tide_time,
                                xmax=tide.high_tide_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="red",
                                linestyle="solid",
                                label=label,
                            )
                        else:
                            ax.hlines(
                                y=tide.max_flood_current,
                                xmin=tide.low_tide_time,
                                xmax=tide.high_tide_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="red",
                                linestyle="solid",
                            )
                    if tide.mean_flood_current is not None:
                        label = "MeanFCS"
                        if label not in legend_handles:
                            legend_handles[label] = ax.hlines(
                                y=tide.mean_flood_current,
                                xmin=tide.low_tide_time,
                                xmax=tide.high_tide_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="red",
                                linestyle="dashed",
                                label=label,
                            )
                        else:
                            ax.hlines(
                                y=tide.mean_flood_current,
                                xmin=tide.low_tide_time,
                                xmax=tide.high_tide_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="red",
                                linestyle="dashed",
                            )
                    if tide.max_ebb_current is not None:
                        label = "MaxECS"
                        if label not in legend_handles:
                            legend_handles[label] = ax.hlines(
                                y=tide.max_ebb_current,
                                xmin=tide.high_tide_time,
                                xmax=tide.low_tide_2_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="green",
                                linestyle="solid",
                                label=label,
                            )
                        else:
                            ax.hlines(
                                y=tide.max_ebb_current,
                                xmin=tide.high_tide_time,
                                xmax=tide.low_tide_2_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="green",
                                linestyle="solid",
                            )
                    if tide.mean_ebb_current is not None:
                        label = "MeanECS"
                        if label not in legend_handles:
                            legend_handles[label] = ax.hlines(
                                y=tide.mean_ebb_current,
                                xmin=tide.high_tide_time,
                                xmax=tide.low_tide_2_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="green",
                                linestyle="dashed",
                                label=label,
                            )
                        else:
                            ax.hlines(
                                y=tide.mean_ebb_current,
                                xmin=tide.high_tide_time,
                                xmax=tide.low_tide_2_time,
                                alpha=0.5,
                                linewidth=0.7,
                                color="green",
                                linestyle="dashed",
                            )

        if plot_slack:
            if hasattr(self, "slack"):
                for index, slack in self.slack.iterrows():
                    if slack["type"] == "to_flood":
                        label = "Se"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.current_speed.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.current_speed.data.name],
                                marker="v",
                                markerfacecolor="none",
                                markeredgecolor="green",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if slack["type"] == "to_ebb":
                        label = "Sf"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                slack[self.current_speed.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                slack[self.current_speed.data.name],
                                marker="^",
                                markerfacecolor="none",
                                markeredgecolor="red",
                                alpha=0.7,
                                linestyle="None",
                            )

        if plot_HWLWs:
            if hasattr(self, "HWLWs"):
                for index, HWLW in self.HWLWs.iterrows():
                    if HWLW["type"] == "HW":
                        label = "Thw"
                        label = "HW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.current_speed.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.current_speed.data.name],
                                marker="^",
                                color="red",
                                alpha=0.7,
                                linestyle="None",
                            )
                    if HWLW["type"] == "LW":
                        label = "Tnw"
                        label = "LW"
                        if label not in legend_handles:
                            legend_handles[label] = ax.plot(
                                index,
                                HWLW[self.current_speed.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                                label=label,
                            )
                        else:
                            ax.plot(
                                index,
                                HWLW[self.current_speed.data.name],
                                marker="v",
                                color="green",
                                alpha=0.7,
                                linestyle="None",
                            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        if ax is None:
            plt.show()

    def _plot(self, variable: Variable, figsize=None):
        if figsize is None:
            figsize = (10, 2.5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(variable.data.index, variable.data.values)
        fig.autofmt_xdate()
        ax.set_ylabel(self._label_txt(data=variable))

        return fig, ax

    def _get_errors(self):
        tide_errors = tuple(
            cls
            for cls in globals().values()
            if isinstance(cls, type)
            and issubclass(cls, TideError)
            and cls is not TideError
        )

        if self.raise_error_types is not None:
            if isinstance(self.raise_error_types, type):
                self.raise_error_types = (self.raise_error_types,)
            errors_to_skip = tuple(
                cls for cls in tide_errors if cls not in self.raise_error_types
            )
        elif self.ignore_error_types is None:
            errors_to_skip = ()
        elif self.ignore_error_types == "all":
            errors_to_skip = tuple(tide_errors) + tuple([Exception])
        elif self.ignore_error_types == "all_tidal":
            errors_to_skip = tuple(tide_errors)
        else:
            if isinstance(self.ignore_error_types, type) and issubclass(
                self.ignore_error_types, Exception
            ):
                errors_to_skip = (self.ignore_error_types,)
            else:
                errors_to_skip = tuple(self.ignore_error_types)

        return tuple(errors_to_skip)

    def _fallswet(self) -> bool:
        """
        Determines if the surface elevation data indicates a 'falls wet' condition.

        Compares consecutive surface elevation data points to check if they are nearly equal,
        based on a specified tolerance. Calculates the percentage of such identical data points
        and compares it against predefined warning and error thresholds.

        Returns:
            bool: True if the percentage of identical data points exceeds the warning threshold;
                False otherwise.

        Raises:
            FallsWetError: If the percentage of identical data points exceeds the error threshold.
        """

        identical_series = (
            self.surface_elevation.data - self.surface_elevation.data.shift()
        ).abs() < self.TOL_WET_ERROR
        identical_count = identical_series.sum()
        total_count = len(self.surface_elevation.data) - 1
        identical_percentage = (
            (identical_count / total_count) * 100 if total_count > 0 else 0
        )
        falls_wet = identical_percentage > self.THRSLD_WET_WARNING * 100

        if identical_percentage > self.THRSLD_WET_ERROR * 100:
            self.fallsdry = None
            self.fallswet = None
            raise FallsWetError(
                f"More than {self.THRSLD_WET_ERROR *100}% of surface elevation values are identical."
            )

        return falls_wet

    def _fallsdry(self) -> bool:
        """
        Determines if the surface elevation data indicates a 'falls dry' condition.

        Checks if there are any NaN values in the surface elevation data.

        Returns:
            bool: True if there are any NaN values; False otherwise.
        """
        falls_dry = np.isnan(self.surface_elevation.data).any()

        return falls_dry

    def _check_data(self):
        """
        Validates the surface elevation data for tidal analysis.

        This method checks if the surface elevation data contains too many NaN values or
        if the total number of data points is below a specified minimum threshold. If either
        condition is met, it raises a NotEnoughWaterError and resets the 'fallsdry' and
        'fallswet' attributes.

        Raises:
            NotEnoughWaterError: If the percentage of NaN values exceeds the threshold
            or if the total data points are less than the minimum required.
        """
        nan_count = np.sum(np.isnan(self.surface_elevation.data))
        total_count = self.surface_elevation.data.size

        if nan_count / total_count > self.THRSLD_NOWATER or total_count < self.MIN_DATA:
            self.fallsdry = False
            self.fallswet = False
            raise NotEnoughWaterError(
                f"More than {self.THRSLD_NOWATER*100}% of surface elevation values are NaN or less then 10 data points."
            )

    @staticmethod
    def _find_peaks_troughs(
        data: pd.DataFrame,
        time_difference: pd.Timedelta | None = None,
        prominence: float | None = None,
    ):
        """
        Finds the peaks and troughs of the input data.

        Args:
            data (pd.DataFrame): The data to find peaks and troughs in.
            time_difference (pd.Timedelta, optional): The time difference between consecutive data points. Defaults to None.
            prominence (float | None): The prominence of the peaks and troughs to find. Defaults to None.

        Returns:
            tuple: A tuple containing two arrays, the first containing the indices of the peaks and the second containing the indices of the troughs.
        """
        if time_difference is not None:
            difference = time_difference // pd.infer_freq(data.index)
        else:
            difference = None
        peaks = find_peaks(data.values, distance=difference, prominence=prominence)[0]
        troughs = find_peaks(-data.values, distance=difference, prominence=prominence)[
            0
        ]

        return peaks, troughs

    def parse_input(
        self,
        surface_elevation: mikeio.DataArray,
        current_speed: mikeio.DataArray | None = None,
        current_direction: mikeio.DataArray | None = None,
        velocity_x: mikeio.DataArray | None = None,
        velocity_y: mikeio.DataArray | None = None,
    ) -> tuple[Variable, Variable, Variable]:
        """
        Parses the input data for tidal analysis.

        This method takes in surface elevation, current speed, current direction,
        velocity x, and velocity y data, and returns the parsed data.

        Args:
            surface_elevation (mikeio.DataArray): Surface elevation data.
            current_speed (mikeio.DataArray, optional): Current speed data. Defaults to None.
            current_direction (mikeio.DataArray, optional): Current direction data. Defaults to None.
            velocity_x (mikeio.DataArray, optional): Velocity x data. Defaults to None.
            velocity_y (mikeio.DataArray, optional): Velocity y data. Defaults to None.

        Returns:
            tuple: A tuple containing the parsed surface elevation, current speed, and current direction data.

        Raises:
            ValueError: If either current speed and current direction, or velocity x and velocity y are not provided.
        """

        surface_elevation = Variable(
            data=surface_elevation.to_dataframe().iloc[:, 0],
            unit=surface_elevation.unit.short_name,
        )

        if current_speed is not None and current_direction is not None:
            current_speed = Variable(
                data=current_speed.to_dataframe().iloc[:, 0],
                unit=current_speed.unit.short_name,
            )
            current_direction = Variable(
                data=current_direction.to_dataframe().iloc[:, 0],
                unit=current_direction.unit.short_name,
            )

        elif velocity_x is not None and velocity_y is not None:
            current_speed, current_direction = self._convert_uv_to_speeddirection(
                velocity_x, velocity_y
            )

            current_speed = Variable(
                data=current_speed.to_dataframe().iloc[:, 0],
                unit=current_speed.unit.short_name,
            )
            current_direction = Variable(
                data=current_direction.to_dataframe().iloc[:, 0],
                unit=current_direction.unit.short_name,
            )

        else:
            raise ValueError(
                "Either current_speed and current_direction, or current_x and current_y must be provided."
            )

        return surface_elevation, current_speed, current_direction

    def _convert_uv_to_speeddirection(
        self, velocity_x, velocity_y
    ) -> tuple[mikeio.DataArray, mikeio.DataArray]:
        """
        Converts u and v velocity components to speed and direction.

        Args:
            velocity_x (mikeio.DataArray): Velocity x data.
            velocity_y (mikeio.DataArray): Velocity y data.

        Returns:
            tuple: A tuple containing the speed and direction DataArrays.
        """
        u = velocity_x.to_numpy()
        v = velocity_y.to_numpy()

        assert (
            velocity_x.unit == EUMUnit.meter_per_sec
        ), "velocity_x must be in meters per second."
        assert (
            velocity_y.unit == EUMUnit.meter_per_sec
        ), "velocity_y must be in meters per second."

        speed = np.sqrt(u**2 + v**2)
        direction = np.mod(90 - np.rad2deg(np.arctan2(v, u)), 360)

        ds_speed = mikeio.DataArray(
            speed,
            time=velocity_x.time,
            item=ItemInfo(
                "Current speed", EUMType.Current_Speed, EUMUnit.meter_per_sec
            ),
            geometry=velocity_x.geometry,
        )
        ds_direction = mikeio.DataArray(
            direction,
            time=velocity_x.time,
            item=ItemInfo(
                "Current direction", EUMType.Current_Direction, EUMUnit.degree
            ),
            geometry=velocity_x.geometry,
        )

        return ds_speed, ds_direction

    def _label_txt(self, data: Variable) -> str:
        return f"{data.data.name} [{data.unit}]"

    def _find_warmup(self, surface_elevation: Variable) -> pd.Timestamp | None:
        """
        Determines the warmup period for the surface elevation data.

        Args:
            surface_elevation (Variable): The surface elevation data.

        Returns:
            pd.Timestamp | None: The timestamp of the first change in surface elevation
            data, or None if the data contains only a single unique value.
        """
        if surface_elevation.data.nunique() == 1:
            return None

        first_diff_idx = surface_elevation.data.ne(
            surface_elevation.data.iloc[0]
        ).idxmax()
        return first_diff_idx
