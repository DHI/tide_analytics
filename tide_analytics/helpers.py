from dataclasses import dataclass, asdict
import pandas as pd


class TideError(Exception):
    pass


class NonAlternatingHWLWsError(TideError):
    pass


class NotEnoughTidesError(TideError):
    pass


class NotEnoughWaterError(TideError):
    pass


class NoSlackPointsFoundError(TideError):
    pass


class NoHWLWsFoundError(TideError):
    pass


class CurrentsToNoisyError(TideError):
    pass


class NonMatchingSlackError(TideError):
    pass


class FallsWetError(TideError):
    pass


class FallsPartiallyDryError(TideError):
    pass


class FallsPartiallyWetError(TideError):
    pass


@dataclass
class TidalErrors:
    FallsPartiallyDryError: bool = False
    FallsPartiallyWetError: bool = False
    FallsWetError: bool = False
    NonAlternatingHWLWsError: bool = False
    NotEnoughWaterError: bool = False
    NotEnoughTidesError: bool = False
    NoSlackPointsFound: bool = False
    NonMatchingSlackError: bool = False
    CurrentsToNoisyError: bool = False
    UnknownError: bool = False

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        active_errors = [
            f"{key}: {value}" for key, value in self.to_dict().items() if value
        ]
        return (
            "\n".join(active_errors)
            if active_errors
            else "TidalErrors: No errors or warnings"
        )


@dataclass
class TidalCharacteristics:
    MHW: float | None = None
    MLW: float | None = None
    MTR: float | None = None
    MTL: float | None = None
    ECD: pd.Timedelta | None = None
    FCD: pd.Timedelta | None = None
    ED: pd.Timedelta | None = None
    FD: pd.Timedelta | None = None
    MAXECS: float | None = None
    MAXFCS: float | None = None
    MEANECS: float | None = None
    MEANFCS: float | None = None

    def __str__(self):
        def format_float(value):
            return (
                f"{value:.4f}" if value is not None and not pd.isna(value) else "None"
            )

        def format_timedelta(value):
            if value is None or pd.isna(value):
                return "None"
            hours, remainder = divmod(value.total_seconds(), 3600)
            minutes = remainder // 60
            return f"{int(hours)} hours {int(minutes)} minutes"

        attributes = [
            f"MHW: {format_float(self.MHW)}",
            f"MLW: {format_float(self.MLW)}",
            f"MTR: {format_float(self.MTR)}",
            f"MTL: {format_float(self.MTL)}",
            f"ED: {format_timedelta(self.ED)}",
            f"FD: {format_timedelta(self.FD)}",
            f"ECD: {format_timedelta(self.ECD)}",
            f"FCD: {format_timedelta(self.FCD)}",
            f"MAXECS: {format_float(self.MAXECS)}",
            f"MAXFCS: {format_float(self.MAXFCS)}",
            f"MEANECS: {format_float(self.MEANECS)}",
            f"MEANFCS: {format_float(self.MEANFCS)}",
        ]
        return "\n".join(attributes)

    def to_dict(self):
        return asdict(self)


@dataclass
class Tide:
    ebb_time: list[pd.Timestamp] | None = None
    flood_time: list[pd.Timestamp] | None = None
    ebb_duration: pd.Timedelta | None = None
    flood_duration: pd.Timedelta | None = None
    ebb_current_time: list[pd.Timestamp] | None = None
    flood_current_time: list[pd.Timestamp] | None = None
    ebb_current_duration: pd.Timedelta | None = None
    flood_current_duration: pd.Timedelta | None = None
    max_flood_current: float | None = None
    max_ebb_current: float | None = None
    mean_flood_current: float | None = None
    mean_ebb_current: float | None = None
    tidal_range: float | None = None
    high_tide: float | None = None
    high_tide_time: pd.Timestamp | None = None
    low_tide: float | None = None
    low_tide_time: pd.Timestamp | None = None
    low_tide_2: float | None = None
    low_tide_2_time: pd.Timestamp | None = None
    mean_tide_level: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Variable:
    data: pd.Series | pd.DataFrame
    unit: str
