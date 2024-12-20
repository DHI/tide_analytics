from dataclasses import dataclass, asdict
import pandas as pd
import mikeio

class TideError(Exception): pass
class NonAlternatingHTLTsError(TideError): pass
class NotEnoughTidesError(TideError): pass
class NotEnoughWaterError(TideError): pass
class NoKenterPointsFoundError(TideError): pass
class NoHTLTsFoundError(TideError): pass
class CurrentsToNoisyError(TideError): pass
class NonMatchingKenterError(TideError): pass
class FallsWetError(TideError): pass

@dataclass
class TidalErrors:
    FallsPartiallyDryWarning: bool = False
    FallsPartiallyWetWarning: bool = False
    FallsWetError: bool = False
    NonAlternatingHTLTsError: bool = False
    NotEnoughWaterError: bool = False
    NotEnoughTidesError: bool = False
    NoKenterPointsFound: bool = False
    NonMatchingKenterError: bool = False
    CurrentsToNoisyError: bool = False
    UnknownError: bool = False
    
    def to_dict(self):
        return asdict(self)

    def __str__(self):
        active_errors = [
            f"{key}: {value}" 
            for key, value in self.to_dict().items() 
            if value
        ]
        return "\n".join(active_errors) if active_errors else "TidalErrors: No errors or warnings"

@dataclass
class TidalCharacteristics:
    MHW: float = None
    MLW: float = None
    TR: float = None
    MTL: float = None
    ECD: pd.Timedelta = None
    FCD: pd.Timedelta = None
    ED: pd.Timedelta = None
    FD: pd.Timedelta = None
    MAXECS: float = None
    MAXFCS: float = None
    MEANECS: float = None
    MEANFCS: float = None

    def __str__(self):
        def format_float(value):
            return f"{value:.4f}" if value is not None and not pd.isna(value) else "None"

        def format_timedelta(value):
            if value is None or pd.isna(value):
                return "None"
            hours, remainder = divmod(value.total_seconds(), 3600)
            minutes = remainder // 60
            return f"{int(hours)} hours {int(minutes)} minutes"

        attributes = [
            f"MHW: {format_float(self.MHW)}",
            f"MLW: {format_float(self.MLW)}",
            f"TR: {format_float(self.TR)}",
            f"ECD: {format_timedelta(self.ECD)}",
            f"FCD: {format_timedelta(self.FCD)}",
            f"ED: {format_timedelta(self.ED)}",
            f"FD: {format_timedelta(self.FD)}",
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
    ebb_time: list[pd.Timestamp, pd.Timestamp] = None
    flood_time: list[pd.Timestamp, pd.Timestamp] = None
    ebb_duration: pd.Timedelta = None
    flood_duration: pd.Timedelta = None
    ebb_current_time: list[pd.Timestamp, pd.Timestamp] = None
    flood_current_time: list[pd.Timestamp, pd.Timestamp] = None
    ebb_current_duration: pd.Timedelta = None
    flood_current_duration: pd.Timedelta = None
    max_flood_current: float = None
    max_ebb_current: float = None
    mean_flood_current: float = None
    mean_ebb_current: float = None
    tidal_range: float = None
    high_tide: float = None
    high_tide_time: pd.Timestamp = None
    low_tide: float = None
    low_tide_time: pd.Timestamp = None
    low_tide_2: float = None
    low_tide_2_time: pd.Timestamp = None
    mean_tide_level: float | None = None

    def to_dict(self):
        return asdict(self)

@dataclass
class Variable:
    data: pd.Series | mikeio.DataArray
    unit: str