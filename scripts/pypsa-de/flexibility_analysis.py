import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
) 

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

def supply_demand(n, region = "DE", interconnectors=True, merge_dist_grid=True, drop_dist_grid=True, buses=None, add_diff_as_import=False):

    """
    Aggregates electricity supply and demand statistics from a PyPSA network for a given region.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing snapshots, buses, and generators.
    region : str, default "DE"
        Two-letter country code prefix for buses to include.
    interconnectors : bool, default True
        Whether to include interconnectors (AC/DC) in the supply/demand.
    merge_dist_grid : bool, default True
        If True, merges low-voltage and AC buses so that the distribution grid only represents losses.
    drop_dist_grid : bool, default True
        If True, removes the electricity distribution grid from supply and demand.
    buses : list[str] or None
        Specific bus names to consider; defaults to all buses in the given region with low-voltage or AC carrier.
    add_diff_as_import : bool, default False
        If True, any difference between total supply and demand is added as "import" (negative difference) or "export" (positive difference).

    Returns
    -------
    electricity_supply : pd.DataFrame
        Supply per carrier aggregated over the selected buses.
    electricity_demand : pd.DataFrame
        Demand per carrier aggregated over the selected buses.

    Notes
    -----
    - Uses n.statistics.supply and n.statistics.withdrawal to compute per-bus and per-carrier values.
    - Multiplies by n.snapshot_weightings.generators to scale per snapshot.
    - Can remove AC/DC interconnectors or the distribution grid according to flags.
    - Optionally computes residual import/export if add_diff_as_import=True.
    - Prints total system supply and demand in TWh (sum / 1e6).
    """

    bus_carrier = ["low voltage", "AC"]

    kwargs = {
        "groupby": n.statistics.groupers.get_name_bus_and_carrier,
        "nice_names": False,
    }

    electricity_supply = (
        n.statistics.supply(
            bus_carrier=bus_carrier, 
            aggregate_time=False, 
            **kwargs
            )
    ).multiply(n.snapshot_weightings.generators) 

    if buses is None:
        buses = n.buses[(n.buses.index.str[:2] == region) & (n.buses.carrier.isin(bus_carrier))].index
    
    electricity_supply = electricity_supply[electricity_supply.index.get_level_values("bus").isin(buses)].groupby("carrier").sum()

    electricity_demand = (
        n.statistics.withdrawal(
            bus_carrier=bus_carrier, 
            aggregate_time=False, 
            **kwargs
            )
    ).multiply(n.snapshot_weightings.generators) 
    electricity_demand = electricity_demand[electricity_demand.index.get_level_values("bus").isin(buses)].groupby("carrier").sum()

    if not interconnectors:
        electricity_supply = electricity_supply.drop(["AC", "DC"], errors="ignore")
        electricity_demand = electricity_demand.drop(["AC", "DC"], errors="ignore")

    if merge_dist_grid:
        # merge AC & low voltage such that electricity distribution grid does only function as a demand representing the grid losses
        electricity_demand.loc["electricity distribution grid losses" , :] = abs(electricity_supply.loc["electricity distribution grid"] - electricity_demand.loc["electricity distribution grid"])
        electricity_demand = electricity_demand.drop("electricity distribution grid", axis=0)
        electricity_supply = electricity_supply.drop("electricity distribution grid", axis=0)
        
    if drop_dist_grid:
        electricity_supply = electricity_supply.drop("electricity distribution grid", errors="ignore")
        electricity_demand = electricity_demand.drop("electricity distribution grid", errors="ignore")

    if add_diff_as_import:
        diff = electricity_supply.sum() - electricity_demand.sum()
        electricity_supply.loc["import"  , : ] = diff.clip(upper=0).abs()
        electricity_demand.loc["export"  , : ] = diff.clip(lower=0)

    return electricity_supply, electricity_demand


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "flexibility_analysis",
            simpl="",
            clusters=27,
            opts="",
            sector_opts="None",
            run="MediumFlex",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons

    # Load networks
    networks = [pypsa.Network(fn) for fn in snakemake.input.networks]
