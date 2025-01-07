import numpy as np
from dataclasses import fields
from mikeio.eum import ItemInfo, EUMUnit, EUMType
import pandas as pd
from mikeio import DataArray
import matplotlib.pyplot as plt
import mikeio
from tqdm.autonotebook import tqdm
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
from sklearn.preprocessing import MinMaxScaler
import dill

from .helpers import TideError, Variable, TidalErrors
from .tidal_series import TidalSeries


class TidalArea:
    def __init__(
        self,
        identifier: str,
        surface_elevation: mikeio.DataArray,
        velocity_x: mikeio.DataArray | None = None,
        velocity_y: mikeio.DataArray | None = None,
        current_speed: mikeio.DataArray | None = None,
        current_direction: mikeio.DataArray | None = None,
        SE_DIF: pd.Timedelta | None = pd.Timedelta("2h"),
        SE_PROM: float | None = None,
        CS_DIF: pd.Timedelta | None = None,
        CS_PROM: float | None = None,
        CD_DIF=pd.Timedelta("1h"),
        THRSLD_WET_ERROR: float = 0.4,
        TOL_WET_ERROR: float = 1e-4,
        THRSLD_WET_WARNING: float = 0.05,
        THRSLD_NOWATER: float = 0.25,
        MIN_DATA: int = 10,
        MIN_TIDES: int = 3,
        MATCH_TOL=None,
        DBSCAN_EPS: float = 10,
        DBSCAN_MINSAMPLES: float = 2,
        ignore_error_types: str | TideError | list[TideError] | None = None,
        raise_error_types: TideError | list[TideError] | None = None,
        skip_n_errors: int = 0,
        verbose: bool = True,
        save_path_parent_folder: Path | None = None,
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

        self.ignore_error_types = ignore_error_types
        self.raise_error_types = raise_error_types

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

        self.skip_n_errors = skip_n_errors
        self.verbose = verbose
        self.save_path_parent_folder = save_path_parent_folder
        self.identifier = identifier

        self.surface_elevation, self.current_speed, self.current_direction = (
            self._parse_input(
                surface_elevation=surface_elevation,
                current_speed=current_speed,
                current_direction=current_direction,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
            )
        )

        self.tidal_characteristics, self.tidal_errors = self.tidal_analysis()

        if self.verbose:
            self.error_support_prompt()
        self.plot_errormap()
        self.plot_statistical_summary()
        self.save()

    def tidal_analysis(self):
        tidal_characteristics_list = []
        tidal_errors_list = []

        skipped_errors = 0
        throw_error = False

        element_range = range(self.surface_elevation.data.geometry.n_elements)
        for element in tqdm(element_range):
            if skipped_errors >= self.skip_n_errors:
                throw_error = True
            try:
                element_se = self.surface_elevation.data.isel(element=element)
                element_cs = self.current_speed.data.isel(element=element)
                element_cd = self.current_direction.data.isel(element=element)
                out = self.tidal_analysis_series(
                    element_se, element_cs, element_cd, throw_error
                )
                tidal_characteristics_list.append(
                    {"element": element, **out.tidal_characteristics.to_dict()}
                )
                tidal_errors_list.append(
                    {"element": element, **out.tidal_errors.to_dict()}
                )
            except Exception as e:
                if throw_error:
                    self.plot_element_on_map(element)
                    print(f"{type(e).__name__}")
                    print(e)
                    self.error_support_prompt(
                        pd.DataFrame(tidal_errors_list)
                        .set_index("element")
                        .sum(axis=0),
                        len(tidal_errors_list),
                    )
                    raise e
                skipped_errors += 1

        tidal_characteristics = pd.DataFrame(tidal_characteristics_list).set_index(
            "element"
        )
        tidal_errors = pd.DataFrame(tidal_errors_list).set_index("element")

        return tidal_characteristics, tidal_errors

    def tidal_analysis_series(
        self, element_se, element_cs, element_cd, error_support
    ) -> TidalSeries:
        out = TidalSeries(
            surface_elevation=element_se,
            current_speed=element_cs,
            current_direction=element_cd,
            SE_DIF=self.SE_DIF,
            SE_PROM=self.SE_PROM,
            CS_DIF=self.CS_DIF,
            CS_PROM=self.CS_PROM,
            MATCH_TOL=self.MATCH_TOL,
            MIN_TIDES=self.MIN_TIDES,
            THRSLD_NOWATER=self.THRSLD_NOWATER,
            THRSLD_WET_ERROR=self.THRSLD_WET_ERROR,
            THRSLD_WET_WARNING=self.THRSLD_WET_WARNING,
            TOL_WET_ERROR=self.TOL_WET_ERROR,
            MIN_DATA=self.MIN_DATA,
            DBSCAN_EPS=self.DBSCAN_EPS,
            DBSCAN_MINSAMPLES=self.DBSCAN_MINSAMPLES,
            ignore_error_types=self.ignore_error_types,
            raise_error_types=self.raise_error_types,
            error_support=error_support,
        )

        return out

    def error_support_prompt(self, tidal_errors_sum=None, investigated_elements=None):
        if tidal_errors_sum is None:
            tidal_errors_sum = self.tidal_errors.sum(axis=0)
            error_rows = tidal_errors_sum.loc[
                tidal_errors_sum.index.str.contains("Error")
            ]
            tidal_errors_sum.loc["Total Errors"] = error_rows.sum(axis=0)
        if investigated_elements is None:
            investigated_elements = self.surface_elevation.data.geometry.n_elements

        all_errors = pd.DataFrame(
            {
                "absolute": tidal_errors_sum[
                    tidal_errors_sum.index.str.contains("Error")
                ],
                "relative": (
                    tidal_errors_sum[tidal_errors_sum.index.str.contains("Error")]
                    / investigated_elements
                ).apply(lambda x: f"{x * 100:.1f} %"),
            }
        )
        print(f"\nError Overview:\n{all_errors.to_string(header=False)}")

        return all_errors

    def save(self):
        if self.save_path_parent_folder is not None:
            if self.save_path_parent_folder is not None:
                self.save_path_parent_folder.mkdir(parents=True, exist_ok=True)
            self.save_dill(self.save_path_parent_folder / f"{self.identifier}.dill")
            self.save_dataframe(
                self.tidal_characteristics,
                path=self.save_path_parent_folder
                / f"tidal_characteristics_{self.identifier}.parquet",
            )
            self.save_dataframe(
                self.tidal_errors,
                path=self.save_path_parent_folder
                / f"tidal_errors_{self.identifier}.parquet",
            )
            self.save_dfsu(self.save_path_parent_folder / f"{self.identifier}.dfsu")

    def save_dataframe(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.suffix:
            path = path.with_suffix(".parquet")

        if path.suffix != ".parquet":
            warnings.warn(f"File suffix changed from {path.suffix} to .parquet.")
            path = path.with_suffix(".parquet")

        df.to_parquet(path)

    def save_dfsu(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.suffix:
            path = path.with_suffix(".dfsu")

        if path.suffix != ".dfsu":
            warnings.warn(f"File suffix changed from {path.suffix} to .dfsu.")
            path = path.with_suffix(".dfsu")

        ds = self.create_output_dataset()

        ds.to_dfs(path)

    def save_dill(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.suffix:
            path = path.with_suffix(".dill")

        if path.suffix != ".dill":
            warnings.warn(f"File suffix changed from {path.suffix} to .dill.")
            path = path.with_suffix(".dill")

        with open(path, "wb") as f:
            dill.dump(self, f)

    def create_output_dataset(self):
        dataset_data = []

        data = self.tidal_characteristics["MHW"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MHW", EUMType.Surface_Elevation, EUMUnit.meter),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MLW"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MLW", EUMType.Surface_Elevation, EUMUnit.meter),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MTL"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MTL", EUMType.Surface_Elevation, EUMUnit.meter),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MTR"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MTR", EUMType.Surface_Elevation, EUMUnit.meter),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MAXECS"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MAXECS", EUMType.Current_Speed, EUMUnit.meter_per_sec),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MAXFCS"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MAXFCS", EUMType.Current_Speed, EUMUnit.meter_per_sec),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MEANECS"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MEANECS", EUMType.Current_Speed, EUMUnit.meter_per_sec),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = self.tidal_characteristics["MEANFCS"].values
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("MEANFCS", EUMType.Current_Speed, EUMUnit.meter_per_sec),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = np.array(
            [td.total_seconds() / 3600 for td in self.tidal_characteristics["ED"]]
        )
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("ED", EUMType.Time, EUMUnit.hour),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = np.array(
            [td.total_seconds() / 3600 for td in self.tidal_characteristics["FD"]]
        )
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("FD", EUMType.Time, EUMUnit.hour),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = np.array(
            [td.total_seconds() / 3600 for td in self.tidal_characteristics["ECD"]]
        )
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("ECD", EUMType.Time, EUMUnit.hour),
                geometry=self.surface_elevation.data.geometry,
            )
        )
        data = np.array(
            [td.total_seconds() / 3600 for td in self.tidal_characteristics["FCD"]]
        )
        dataset_data.append(
            DataArray(
                data,
                time=str(self.surface_elevation.data.time.mean()),
                item=ItemInfo("FCD", EUMType.Time, EUMUnit.hour),
                geometry=self.surface_elevation.data.geometry,
            )
        )

        ds = mikeio.Dataset(dataset_data).copy()

        return ds

    def _parse_input(
        self,
        surface_elevation: mikeio.DataArray,
        current_speed: mikeio.DataArray | None = None,
        current_direction: mikeio.DataArray | None = None,
        velocity_x: mikeio.DataArray | None = None,
        velocity_y: mikeio.DataArray | None = None,
    ):
        surface_elevation_out = Variable(
            data=surface_elevation, unit=surface_elevation.unit.short_name
        )

        if current_speed is not None and current_direction is not None:
            if current_direction.unit == EUMUnit.radian:
                current_direction = current_direction * 180 / np.pi
                current_direction.item = ItemInfo(
                    "Current direction", current_direction.item.type, EUMUnit.degree
                )
            current_speed_out = Variable(
                data=current_speed, unit=current_speed.unit.short_name
            )
            current_direction_out = Variable(
                data=current_direction, unit=current_direction.unit.short_name
            )

        elif velocity_x is not None and velocity_y is not None:
            current_speed, current_direction = self._convert_uv_to_speeddirection(
                velocity_x, velocity_y
            )

            current_speed_out = Variable(
                data=current_speed, unit=current_speed.unit.short_name
            )
            current_direction_out = Variable(
                data=current_direction, unit=current_direction.unit.short_name
            )

        else:
            raise ValueError(
                "Either current_speed and current_direction, or velocity_x and velocity_y must be provided."
            )

        return surface_elevation_out, current_speed_out, current_direction_out

    def _convert_uv_to_speeddirection(
        self, velocity_x, velocity_y
    ) -> tuple[mikeio.DataArray, mikeio.DataArray]:
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
                "Current speed", EUMType.Current_Direction, EUMUnit.meter_per_sec
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

    def plot_element_on_map(
        self, element, figsize: tuple[float, float] = (12, 12), show=True
    ):
        el_coords = self.surface_elevation.data.geometry.element_coordinates[element]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.surface_elevation.data.geometry.plot(ax=ax)
        ax.plot(el_coords[0], el_coords[1], marker="*", markersize=10, c="r")
        ax.set_title("")

        if show:
            plt.show()

        return ax

    def plot_errormap(self, figsize: tuple[float, float] = (12, 12)):
        if self.tidal_errors is None:
            raise ValueError(
                "Tidal errors have not been calculated yet. Please run the tidal analysis first."
            )

        error_to_color = {
            field.name: idx for idx, field in enumerate(fields(TidalErrors))
        }
        data = np.full(len(self.tidal_errors), np.nan)

        present_errors = []
        for error, color_idx in error_to_color.items():
            if error in self.tidal_errors.columns:
                idx = np.where(self.tidal_errors[error])
                if len(idx[0]) > 0:
                    present_errors.append(error)
                    data[idx] = color_idx
            else:
                raise ValueError(
                    f"Error {error} not found in tidal_errors. Please add it."
                )

        cmap = plt.get_cmap("tab20")
        color_indices = [error_to_color[error] for error in present_errors]
        colors = [cmap(i) for i in color_indices]
        cmap = ListedColormap(colors)

        nan_indices = np.isnan(data)
        sorted_arr = np.unique(data[~nan_indices])
        arr_sorted_indices = np.searchsorted(sorted_arr, data[~nan_indices])
        data = np.full_like(data, np.nan, dtype=float)
        data[~nan_indices] = arr_sorted_indices

        dataarray = DataArray(
            data,
            time=str(self.surface_elevation.data.time.mean()),
            item=ItemInfo("Errors", EUMType.Status, EUMUnit.__),
            geometry=self.surface_elevation.data.geometry,
        )

        fig, ax = plt.subplots(figsize=figsize)
        dataarray.plot(ax=ax, cmap=cmap, add_colorbar=False)
        ax.set_title("Error Map")

        handles = [
            plt.Line2D(
                [0], [0], marker="s", color="w", markerfacecolor=cmap(i), markersize=10
            )
            for i, _ in enumerate(present_errors)
        ]
        ax.legend(
            handles=handles,
            labels=present_errors,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
        )

        if self.save_path_parent_folder is not None:
            self.save_path_parent_folder.mkdir(parents=True, exist_ok=True)
            save_path = (
                self.save_path_parent_folder / f"tidal_errors_{self.identifier}.png"
            )
            fig.savefig(save_path, dpi=600)

        if self.verbose:
            plt.show()

        return ax

    def plot_statistical_summary(self, figsize: tuple[float, float] = (16, 6)):
        df = self.tidal_characteristics.copy()

        float_cols = [
            "MHW",
            "MLW",
            "MTR",
            "MTL",
            "MAXECS",
            "MAXFCS",
            "MEANECS",
            "MEANFCS",
        ]
        timedelta_cols = ["ECD", "FCD", "ED", "FD"]
        df = df[float_cols + timedelta_cols]

        df_timedelta_seconds = df[timedelta_cols].apply(
            lambda col: col.dt.total_seconds() / 3600
        )

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(
            pd.concat([df[float_cols], df_timedelta_seconds], axis=1)
        )
        scaled_df = pd.DataFrame(scaled_data, columns=float_cols + timedelta_cols)

        assert all(
            df.columns == scaled_df.columns
        ), "The columns of the dataframe and the scaled dataframe are not the same"

        fig, ax = plt.subplots(figsize=figsize)
        _, boxplot = scaled_df.boxplot(
            grid=False,
            return_type="both",
            notch=True,
            patch_artist=True,
            boxprops={"edgecolor": "black"},
            flierprops={
                "marker": ".",
                "markersize": 7,
                "markerfacecolor": "grey",
                "markeredgecolor": "None",
                "alpha": 0.5,
            },
        )
        plt.xlabel("Tidal Characteristic")
        plt.ylabel("MinMax Scaled Data")

        outliers = [flier.get_ydata() for flier in boxplot["fliers"]]
        outliers = [len(column) / len(df) * 100 for column in outliers]
        df[timedelta_cols] = df_timedelta_seconds
        means = df.mean()
        maxs = df.max()
        mins = df.min()
        nans = df.isna().sum() / len(df) * 100

        x_positions = range(1, len(df.columns) + 1)
        for i, col in enumerate(df.columns):
            stats_text = f"Mean:     {means.iloc[i]:.2f}\nMax:       {maxs.iloc[i]:.2f}\nMin:        {mins.iloc[i]:.2f}\nOutliers: {outliers[i]:.1f} %\nNo Data: {nans.iloc[i]:.1f} %"
            plt.text(
                x_positions[i],
                1.3,  # Position below the x-axis
                stats_text,
                ha="center",
                va="top",
                fontsize=10,
                multialignment="left",
                bbox=dict(
                    boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.5
                ),
                transform=plt.gca().transData,
            )

        plt.tight_layout()

        if self.save_path_parent_folder is not None:
            if self.save_path_parent_folder is not None:
                self.save_path_parent_folder.mkdir(parents=True, exist_ok=True)
            save_path = (
                self.save_path_parent_folder
                / f"statistical_summary_{self.identifier}.png"
            )
            fig.savefig(save_path, dpi=600)

        if self.verbose:
            plt.show()

        return ax
