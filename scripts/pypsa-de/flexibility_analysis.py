import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from flexibility_utils import tech_groups

logger = logging.getLogger(__name__)


def calc_supply_demand(
    n,
    bus_carrier=["low voltage", "AC", "EV battery"],
    region="DE",
    energy=True,
    interconnectors=True,
    merge_dist_grid=True,
    drop_dist_grid=True,
    buses=None,
    add_diff_as_import=False,
):
    """
    Aggregates supply and demand statistics from a PyPSA network for a given region or list of buses.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing snapshots, buses, and generators.
    bus_carrier : list[str], default ["low voltage", "AC"]
        Carrier for which supply and demand is calculated.
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
    supply : pd.DataFrame
        Supply per carrier aggregated over the selected buses.
    demand : pd.DataFrame
        Demand per carrier aggregated over the selected buses.
    """

    if buses is None:
        buses = n.buses[
            (n.buses.index.str[:2] == region) & (n.buses.carrier.isin(bus_carrier))
        ].index

    kwargs = {
        "groupby": n.statistics.groupers.get_name_bus_and_carrier,
        "nice_names": False,
    }

    supply = n.statistics.supply(
        bus_carrier=bus_carrier, aggregate_time=False, **kwargs
    )
    demand = n.statistics.withdrawal(
        bus_carrier=bus_carrier, aggregate_time=False, **kwargs
    )

    if energy:
        supply = supply.multiply(n.snapshot_weightings.generators)
        demand = demand.multiply(n.snapshot_weightings.generators)

    supply = (
        supply[supply.index.get_level_values("bus").isin(buses)]
        .groupby("carrier")
        .sum()
    )
    demand = (
        demand[demand.index.get_level_values("bus").isin(buses)]
        .groupby("carrier")
        .sum()
    )

    if not interconnectors:
        supply = supply.drop(["AC", "DC"], errors="ignore")
        demand = demand.drop(["AC", "DC"], errors="ignore")

    if "land transport EV" in demand.index:
        charge_efficiency = demand.loc["land transport EV"].sum() / demand.loc["BEV charger"].sum()
        supply = supply.drop("BEV charger", errors="ignore")
        demand = demand.drop("BEV charger", errors="ignore")
        demand.loc["land transport EV"] = demand.loc["land transport EV"] / charge_efficiency

    if merge_dist_grid:
        # merge AC & low voltage such that electricity distribution grid does only function as a demand representing the grid losses
        demand.loc["electricity distribution grid losses", :] = abs(
            supply.loc["electricity distribution grid"]
            - demand.loc["electricity distribution grid"]
        )
        demand = demand.drop("electricity distribution grid", axis=0)
        supply = supply.drop("electricity distribution grid", axis=0)

    if drop_dist_grid:
        supply = supply.drop("electricity distribution grid", errors="ignore")
        demand = demand.drop("electricity distribution grid", errors="ignore")

    if add_diff_as_import:
        diff = supply.sum() - demand.sum()
        supply.loc["import", :] = diff.clip(upper=0).abs()
        demand.loc["export", :] = diff.clip(lower=0)

    # drop empty rows
    supply = supply[supply.any(axis=1)]
    demand = demand[demand.any(axis=1)]

    if not np.isclose(supply.sum().sum(), demand.sum().sum()):
        raise ValueError(
            f"Supply and demand do not match: {supply.sum().sum()} vs {demand.sum().sum()}"
        )

    return supply, demand


def calc_residual_load(
    supply,
    demand,
    non_dispatchable_supply_carriers=[
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "solar-hsat",
        "solar rooftop",
        "ror",
    ],
    non_dispatchable_demand_carriers=[
        "electricity",
        "agriculture electricity",
        "industry electricity",
        "agriculture machinery electric",
        "land transport EV",
        'rural air heat pump',
        'rural ground heat pump',
        'urban decentral air heat pump',
    ],
):
    """
    Calculate residual load from supply and demand data.

    Parameters
    ----------
    supply : pd.DataFrame
        Supply data with carriers as index and time as columns
    demand : pd.DataFrame
        Demand data with carriers as index and time as columns
    non_dispatchable_supply_carriers : list, default renewable carriers
        List of non-dispatchable supply carrier names (renewables)
    non_dispatchable_demand_carriers : list, default electricity carriers
        List of demand carrier names to include

    Returns
    -------
    pd.Series
        Residual load time series (demand - renewable generation)
    """

    # Get available non-dispatchable supply carriers
    nds_available = [
        carrier
        for carrier in non_dispatchable_supply_carriers
        if carrier in supply.index
    ]

    # Get available demand carriers
    ndd_available = [
        carrier
        for carrier in non_dispatchable_demand_carriers
        if carrier in demand.index
    ]

    # Calculate totals
    if nds_available:
        non_dispatchable_gen = supply.loc[nds_available].sum(axis=0)
    else:
        non_dispatchable_gen = pd.Series(0, index=supply.columns)

    if ndd_available:
        non_dispatchable_con = demand.loc[ndd_available].sum(axis=0)
    else:
        non_dispatchable_con = pd.Series(0, index=demand.columns)

    # Calculate residual load
    residual_load = non_dispatchable_con - non_dispatchable_gen

    return residual_load


def calc_flexibility_needs(
    residual_load: pd.Series, granularity: str = "all"
) -> pd.DataFrame:
    """
    Calculate flexibility needs at different granularities (daily, weekly, annual)
    based on Artelys methodology.

    Parameters
    ----------
    residual_load : pd.Series
        Time series of residual load (indexed by datetime, in MWh per hour or step).
    granularity : str, optional
        Which granularity to compute: "daily", "weekly", "annual", or "all" (default).

    Returns
    -------
    pd.DataFrame
        DataFrame with total annual flexibility needs (TWh/year) for selected granularities.
    """
    results = {}

    # --- Precompute averages ---
    daily_avg = residual_load.resample("D").transform("mean")
    weekly_avg = residual_load.resample("W").transform("mean")
    annual_avg = residual_load.resample("YS").transform("mean")

    # --- Daily ---
    if granularity in ["daily", "all"]:
        dev = (residual_load - daily_avg).abs()
        flex = 0.5 * dev.sum()
        results["daily"] = flex / 1e6  # TWh/year

    # --- Weekly ---
    if granularity in ["weekly", "all"]:
        dev = (daily_avg - weekly_avg).abs()
        flex = 0.5 * dev.sum()
        results["weekly"] = flex / 1e6

    # --- Annual ---
    if granularity in ["annual", "all"]:
        dev = (weekly_avg - annual_avg).abs()
        flex = 0.5 * dev.sum()
        results["annual"] = flex / 1e6

    return pd.DataFrame.from_dict(
        results, orient="index", columns=["Flexibility (TWh/year)"]
    )


def calc_flexibility_needs_with_monthly(
    residual_load: pd.Series, granularity: str = "all"
) -> pd.DataFrame:
    """
    Calculate flexibility needs at different granularities (daily, weekly, monthly, annual)
    based on Artelys methodology.

    Parameters
    ----------
    residual_load : pd.Series
        Time series of residual load (indexed by datetime, in MWh per hour or step).
    granularity : str, optional
        Which granularity to compute: "daily", "weekly", "monthly", "annual", or "all" (default).

    Returns
    -------
    pd.DataFrame
        DataFrame with total annual flexibility needs (TWh/year) for selected granularities.
    """

    results = {}

    # --- Precompute averages ---
    daily_avg = residual_load.resample("D").transform("mean")
    weekly_avg = residual_load.resample("W").transform("mean")
    monthly_avg = residual_load.resample("MS").transform("mean")
    annual_avg = residual_load.resample("YS").transform("mean")

    # --- Daily ---
    if granularity in ["daily", "all"]:
        dev = (residual_load - daily_avg).abs()
        flex = 0.5 * dev.sum()
        results["daily"] = flex.sum() / 1e6  # TWh/year

    # --- Weekly ---
    if granularity in ["weekly", "all"]:
        dev = (daily_avg - weekly_avg).abs()
        flex = 0.5 * dev.sum()
        results["weekly"] = flex.sum() / 1e6

    # --- Monthly ---
    if granularity in ["monthly", "all"]:
        dev = (weekly_avg - monthly_avg).abs()
        flex = 0.5 * dev.sum()
        results["monthly"] = flex.sum() / 1e6

    # --- Annual ---
    if granularity in ["annual", "all"]:
        dev = (monthly_avg - annual_avg).abs()
        flex = 0.5 * dev.sum()
        results["annual"] = flex.sum() / 1e6

    return pd.DataFrame.from_dict(
        results, orient="index", columns=["Flexibility (TWh/year)"]
    )


def calc_flexibility_contributions(
    electricity_supply: pd.DataFrame,
    electricity_demand: pd.DataFrame,
    non_dispatchable_supply_carriers=[
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "solar-hsat",
        "solar rooftop",
        "ror",
    ],
    non_dispatchable_demand_carriers=[
        "electricity",
        "agriculture electricity",
        "industry electricity",
        "agriculture machinery electric",
        "land transport EV",
        'rural air heat pump',
        'rural ground heat pump',
        'urban decentral air heat pump',
    ],
    granularity: str = "all",
    analyze: str = "both",
    print_info: bool = False,
) -> tuple or pd.DataFrame:
    """
    Calculate technology contributions to flexibility needs using correlation-based method
    following Artelys methodology. Returns both causes and solutions of flexibility needs.

    Parameters
    ----------
    electricity_supply : pd.DataFrame
        DataFrame with carriers as index and timestamps as columns (supply is positive).
    electricity_demand : pd.DataFrame
        DataFrame with carriers as index and timestamps as columns (demand is positive).
    non_dispatchable_supply_carriers : list, default renewable carriers
        List of non-dispatchable supply carrier names that cause flexibility needs.
    non_dispatchable_demand_carriers : list, default electricity carriers
        List of demand carrier names that cause flexibility needs.
    granularity : str, optional
        Which granularity to compute: "daily", "weekly", "annual", or "all" (default).
    analyze : str, optional
        What to analyze: "flexible", "inflexible", or "both" (default).
        - "flexible": Only analyze dispatchable technologies (solutions)
        - "inflexible": Only analyze non-dispatchable technologies (causes)
        - "both": Analyze both types (returns tuple of DataFrames)

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, pd.DataFrame)
        If analyze="flexible": flexible_contributions DataFrame
        If analyze="inflexible": inflexible_contributions DataFrame
        If analyze="both": (flexible_contributions, inflexible_contributions)
        - flexible_contributions: Technologies that can respond to flexibility needs (positive = helps)
        - inflexible_contributions: Technologies that cause flexibility needs (negative = causes needs)
    """

    # Validate analyze parameter
    if analyze not in ["flexible", "inflexible", "both"]:
        raise ValueError("analyze must be 'flexible', 'inflexible', or 'both'")

    # Calculate residual load (default specification)
    residual_load = calc_residual_load(
        electricity_supply,
        electricity_demand,
        non_dispatchable_supply_carriers,
        non_dispatchable_demand_carriers,
    )

    results_flexible = {}
    results_inflexible = {}

    # Separate flexible and inflexible technologies
    all_flexible = {}
    all_inflexible = {}

    # Process all technologies in one loop
    for df, prefix, non_dispatch_list in [
        (electricity_supply, "Supply_", non_dispatchable_supply_carriers),
        (electricity_demand, "Demand_", non_dispatchable_demand_carriers),
    ]:
        for carrier in df.index:
            # Create tech series (negative for demand to represent consumption)
            multiplier = -1 if prefix == "Demand_" else 1
            tech_series = multiplier * pd.Series(
                df.loc[carrier].values,
                index=residual_load.index,
                name=f"{prefix}{carrier}",
            )

            # Classify as flexible or inflexible
            if carrier in non_dispatch_list:
                all_inflexible[f"{prefix}{carrier}"] = tech_series
            else:
                all_flexible[f"{prefix}{carrier}"] = tech_series

    if print_info:
        logger.info(f"Flexible technologies (can respond): {len(all_flexible)}")
        logger.info(f"Inflexible technologies (cause needs): {len(all_inflexible)}")
        logger.info("")

    # Check what to analyze
    analyze_flexible = analyze in ["flexible", "both"]
    analyze_inflexible = analyze in ["inflexible", "both"]

    if analyze_flexible and not all_flexible:
        logger.info("Warning: No flexible technologies found!")
    if analyze_inflexible and not all_inflexible:
        logger.info("Warning: No inflexible technologies found!")

    # Calculate contributions for each granularity
    granularities_to_calc = (
        ["daily", "weekly", "annual"] if granularity == "all" else [granularity]
    )

    for gran in granularities_to_calc:
        if print_info:
            logger.info(f"=== {gran.upper()} FLEXIBILITY ANALYSIS ===")

        # Calculate flexibility curve and sign for this granularity
        if gran == "daily":
            avg_residual = residual_load.resample("D").transform("mean")
            flexibility_curve_residual = residual_load - avg_residual

        elif gran == "weekly":
            daily_avg_residual = residual_load.resample("D").transform("mean")
            weekly_avg_residual = residual_load.resample("W").transform("mean")
            flexibility_curve_residual = daily_avg_residual - weekly_avg_residual

        elif gran == "annual":
            weekly_avg_residual = residual_load.resample("W").transform("mean")
            annual_avg_residual = residual_load.mean()
            flexibility_curve_residual = weekly_avg_residual - annual_avg_residual

        flex_sign = np.sign(flexibility_curve_residual)
        total_flex_needs = 0.5 * flexibility_curve_residual.abs().sum() / 1e6

        if print_info:
            logger.info(
                f"Total {gran} flexibility needs: {total_flex_needs:.3f} TWh/year"
            )

        # Helper function to calculate contributions for a set of technologies
        def calc_contributions(tech_dict, gran, flex_sign):
            contributions = {}
            for tech_name, tech_profile in tech_dict.items():
                if gran == "daily":
                    avg_tech = tech_profile.resample("D").transform("mean")
                    flexibility_curve_tech = tech_profile - avg_tech

                elif gran == "weekly":
                    daily_avg_tech = tech_profile.resample("D").transform("mean")
                    weekly_avg_tech = tech_profile.resample("W").transform("mean")
                    flexibility_curve_tech = daily_avg_tech - weekly_avg_tech

                elif gran == "annual":
                    weekly_avg_tech = tech_profile.resample("W").transform("mean")
                    annual_avg_tech = tech_profile.mean()
                    flexibility_curve_tech = weekly_avg_tech - annual_avg_tech

                contribution = 0.5 * (flexibility_curve_tech * flex_sign).sum() / 1e6
                contributions[tech_name] = contribution
            return contributions

        # Calculate contributions based on analyze parameter
        if analyze_flexible:
            results_flexible[gran] = calc_contributions(all_flexible, gran, flex_sign)
        if analyze_inflexible:
            results_inflexible[gran] = calc_contributions(
                all_inflexible, gran, flex_sign
            )

        # Verification
        if analyze == "both":
            sum_flexible = sum(results_flexible[gran].values())
            sum_inflexible = sum(results_inflexible[gran].values())
            total_sum = sum_flexible + sum_inflexible

            if print_info:
                logger.info(
                    f"Flexible technology contributions sum: {sum_flexible:.3f} TWh/year"
                )
                logger.info(
                    f"Inflexible technology contributions sum: {sum_inflexible:.3f} TWh/year"
                )
                logger.info(f"Combined sum: {total_sum:.3f} TWh/year")
                logger.info(
                    f"Difference from total needs: {abs(total_flex_needs - total_sum):.6f} TWh/year"
                )

            # Check if inflexible contributions are negative (causing flexibility needs)
            if sum_inflexible < 0:
                if print_info:
                    logger.info(
                        f"✓ Inflexible technologies cause flexibility needs (negative sum: {sum_inflexible:.3f})"
                    )
            else:
                if print_info:
                    logger.info(
                        f"⚠ Warning: Inflexible technologies have positive sum: {sum_inflexible:.3f}"
                    )

        elif analyze == "flexible" and print_info:
            sum_flexible = sum(results_flexible[gran].values())
            logger.info(
                f"Flexible technology contributions sum: {sum_flexible:.3f} TWh/year"
            )

        elif analyze == "inflexible" and print_info:
            sum_inflexible = sum(results_inflexible[gran].values())
            logger.info(
                f"Inflexible technology contributions sum: {sum_inflexible:.3f} TWh/year"
            )
            if sum_inflexible < 0:
                logger.info(
                    f"✓ Inflexible technologies cause flexibility needs (negative sum: {sum_inflexible:.3f})"
                )

        if print_info:
            logger.info("")

    # Convert to DataFrames for output
    def create_dataframe(results_dict, granularity):
        if granularity == "all":
            data = []
            for gran, tech_dict in results_dict.items():
                for tech, contrib in tech_dict.items():
                    data.append(
                        {
                            "Granularity": gran,
                            "Technology": tech,
                            "Contribution (TWh/year)": contrib,
                        }
                    )
            return pd.DataFrame(data).set_index(["Granularity", "Technology"])
        else:
            return pd.DataFrame.from_dict(
                results_dict[granularity],
                orient="index",
                columns=["Contribution (TWh/year)"],
            )

    # Return appropriate DataFrames based on analyze parameter
    if analyze == "flexible":
        return create_dataframe(results_flexible, granularity)
    elif analyze == "inflexible":
        return create_dataframe(results_inflexible, granularity)
    else:  # analyze == "both"
        flexible_df = create_dataframe(results_flexible, granularity)
        inflexible_df = create_dataframe(results_inflexible, granularity)
        return flexible_df, inflexible_df


def calc_flexibility_contributions_with_monthly(
    electricity_supply: pd.DataFrame,
    electricity_demand: pd.DataFrame,
    non_dispatchable_supply_carriers=[
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "solar-hsat",
        "solar rooftop",
        "ror",
    ],
    non_dispatchable_demand_carriers=[
        "electricity",
        "agriculture electricity",
        "industry electricity",
        "agriculture machinery electric",
        "land transport EV",
        'rural air heat pump',
        'rural ground heat pump',
        'urban decentral air heat pump',
    ],
    granularity: str = "all",
    analyze: str = "both",
    print_info: bool = False,
) -> tuple or pd.DataFrame:
    """
    Calculate technology contributions to flexibility needs using correlation-based method
    following Artelys methodology. Returns both causes and solutions of flexibility needs.

    Parameters
    ----------
    electricity_supply : pd.DataFrame
        DataFrame with carriers as index and timestamps as columns (supply is positive).
    electricity_demand : pd.DataFrame
        DataFrame with carriers as index and timestamps as columns (demand is positive).
    non_dispatchable_supply_carriers : list, default renewable carriers
        List of non-dispatchable supply carrier names that cause flexibility needs.
    non_dispatchable_demand_carriers : list, default electricity carriers
        List of demand carrier names that cause flexibility needs.
    granularity : str, optional
        Which granularity to compute: "daily", "weekly", "monthly", "annual", or "all" (default).
    analyze : str, optional
        What to analyze: "flexible", "inflexible", or "both" (default).
        - "flexible": Only analyze dispatchable technologies (solutions)
        - "inflexible": Only analyze non-dispatchable technologies (causes)
        - "both": Analyze both types (returns tuple of DataFrames)

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, pd.DataFrame)
        If analyze="flexible": flexible_contributions DataFrame
        If analyze="inflexible": inflexible_contributions DataFrame
        If analyze="both": (flexible_contributions, inflexible_contributions)
        - flexible_contributions: Technologies that can respond to flexibility needs (positive = helps)
        - inflexible_contributions: Technologies that cause flexibility needs (negative = causes needs)
    """

    # Validate analyze parameter
    if analyze not in ["flexible", "inflexible", "both"]:
        raise ValueError("analyze must be 'flexible', 'inflexible', or 'both'")

    # Calculate residual load
    residual_load = calc_residual_load(
        electricity_supply,
        electricity_demand,
        non_dispatchable_supply_carriers,
        non_dispatchable_demand_carriers,
    )

    results_flexible = {}
    results_inflexible = {}

    # Separate flexible and inflexible technologies
    all_flexible = {}
    all_inflexible = {}

    # Process all technologies in one loop
    for df, prefix, non_dispatch_list in [
        (electricity_supply, "Supply_", non_dispatchable_supply_carriers),
        (electricity_demand, "Demand_", non_dispatchable_demand_carriers),
    ]:
        for carrier in df.index:
            # Create tech series (negative for demand to represent consumption)
            multiplier = -1 if prefix == "Demand_" else 1
            tech_series = multiplier * pd.Series(
                df.loc[carrier].values,
                index=residual_load.index,
                name=f"{prefix}{carrier}",
            )

            # Classify as flexible or inflexible
            if carrier in non_dispatch_list:
                all_inflexible[f"{prefix}{carrier}"] = tech_series
            else:
                all_flexible[f"{prefix}{carrier}"] = tech_series

    if print_info:
        logger.info(f"Flexible technologies (can respond): {len(all_flexible)}")
        logger.info(f"Inflexible technologies (cause needs): {len(all_inflexible)}")
        logger.info("")

    # Check what to analyze
    analyze_flexible = analyze in ["flexible", "both"]
    analyze_inflexible = analyze in ["inflexible", "both"]

    if analyze_flexible and not all_flexible:
        logger.info("Warning: No flexible technologies found!")
    if analyze_inflexible and not all_inflexible:
        logger.info("Warning: No inflexible technologies found!")

    # Calculate contributions for each granularity
    granularities_to_calc = (
        ["daily", "weekly", "monthly", "annual"]
        if granularity == "all"
        else [granularity]
    )

    for gran in granularities_to_calc:
        if print_info:
            logger.info(f"=== {gran.upper()} FLEXIBILITY ANALYSIS ===")

        # Calculate flexibility curve and sign for this granularity
        if gran == "daily":
            avg_residual = residual_load.resample("D").transform("mean")
            flexibility_curve_residual = residual_load - avg_residual

        elif gran == "weekly":
            daily_avg_residual = residual_load.resample("D").transform("mean")
            weekly_avg_residual = residual_load.resample("W").transform("mean")
            flexibility_curve_residual = daily_avg_residual - weekly_avg_residual

        elif gran == "monthly":
            weekly_avg_residual = residual_load.resample("W").transform("mean")
            monthly_avg_residual = residual_load.resample("MS").transform("mean")
            flexibility_curve_residual = weekly_avg_residual - monthly_avg_residual

        elif gran == "annual":
            monthly_avg_residual = residual_load.resample("MS").transform("mean")
            annual_avg_residual = residual_load.mean()
            flexibility_curve_residual = monthly_avg_residual - annual_avg_residual

        flex_sign = np.sign(flexibility_curve_residual)
        total_flex_needs = 0.5 * flexibility_curve_residual.abs().sum() / 1e6

        if print_info:
            logger.info(
                f"Total {gran} flexibility needs: {total_flex_needs:.3f} TWh/year"
            )

        # Helper function to calculate contributions for a set of technologies
        def calc_contributions(tech_dict, gran, flex_sign):
            contributions = {}
            for tech_name, tech_profile in tech_dict.items():
                if gran == "daily":
                    avg_tech = tech_profile.resample("D").transform("mean")
                    flexibility_curve_tech = tech_profile - avg_tech

                elif gran == "weekly":
                    daily_avg_tech = tech_profile.resample("D").transform("mean")
                    weekly_avg_tech = tech_profile.resample("W").transform("mean")
                    flexibility_curve_tech = daily_avg_tech - weekly_avg_tech

                elif gran == "monthly":
                    weekly_avg_tech = tech_profile.resample("W").transform("mean")
                    monthly_avg_tech = tech_profile.resample("MS").transform("mean")
                    flexibility_curve_tech = weekly_avg_tech - monthly_avg_tech

                elif gran == "annual":
                    monthly_avg_tech = tech_profile.resample("MS").transform("mean")
                    annual_avg_tech = tech_profile.mean()
                    flexibility_curve_tech = monthly_avg_tech - annual_avg_tech

                contribution = 0.5 * (flexibility_curve_tech * flex_sign).sum() / 1e6
                contributions[tech_name] = contribution
            return contributions

        # Calculate contributions based on analyze parameter
        if analyze_flexible:
            results_flexible[gran] = calc_contributions(all_flexible, gran, flex_sign)
        if analyze_inflexible:
            results_inflexible[gran] = calc_contributions(
                all_inflexible, gran, flex_sign
            )

        # Verification
        if analyze == "both":
            sum_flexible = sum(results_flexible[gran].values())
            sum_inflexible = sum(results_inflexible[gran].values())
            total_sum = sum_flexible + sum_inflexible

            if print_info:
                logger.info(
                    f"Flexible technology contributions sum: {sum_flexible:.3f} TWh/year"
                )
                logger.info(
                    f"Inflexible technology contributions sum: {sum_inflexible:.3f} TWh/year"
                )
                logger.info(f"Combined sum: {total_sum:.3f} TWh/year")
                logger.info(
                    f"Difference from total needs: {abs(total_flex_needs - total_sum):.6f} TWh/year"
                )

            # Check if inflexible contributions are negative (causing flexibility needs)
            if sum_inflexible < 0:
                if print_info:
                    logger.info(
                        f"✓ Inflexible technologies cause flexibility needs (negative sum: {sum_inflexible:.3f})"
                    )
            else:
                if print_info:
                    logger.info(
                        f"⚠ Warning: Inflexible technologies have positive sum: {sum_inflexible:.3f}"
                    )

        elif analyze == "flexible" and print_info:
            sum_flexible = sum(results_flexible[gran].values())
            logger.info(
                f"Flexible technology contributions sum: {sum_flexible:.3f} TWh/year"
            )

        elif analyze == "inflexible" and print_info:
            sum_inflexible = sum(results_inflexible[gran].values())
            logger.info(
                f"Inflexible technology contributions sum: {sum_inflexible:.3f} TWh/year"
            )
            if sum_inflexible < 0:
                logger.info(
                    f"✓ Inflexible technologies cause flexibility needs (negative sum: {sum_inflexible:.3f})"
                )

        if print_info:
            logger.info("")

    # Convert to DataFrames for output
    def create_dataframe(results_dict, granularity):
        if granularity == "all":
            data = []
            for gran, tech_dict in results_dict.items():
                for tech, contrib in tech_dict.items():
                    data.append(
                        {
                            "Granularity": gran,
                            "Technology": tech,
                            "Contribution (TWh/year)": contrib,
                        }
                    )
            return pd.DataFrame(data).set_index(["Granularity", "Technology"])
        else:
            return pd.DataFrame.from_dict(
                results_dict[granularity],
                orient="index",
                columns=["Contribution (TWh/year)"],
            )

    # Return appropriate DataFrames based on analyze parameter
    if analyze == "flexible":
        return create_dataframe(results_flexible, granularity)
    elif analyze == "inflexible":
        return create_dataframe(results_inflexible, granularity)
    else:  # analyze == "both"
        flexible_df = create_dataframe(results_flexible, granularity)
        inflexible_df = create_dataframe(results_inflexible, granularity)
        return flexible_df, inflexible_df


def expand_to_1h(df: pd.DataFrame, unit: str, tol: float = 1e-9) -> pd.DataFrame:
    """
    Convert a DataFrame with any time resolution to 1-hourly resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with carriers as index and snapshots as columns
    unit : str
        Unit of the input data (required):
        - 'energy' or 'MWh': input is energy, will be converted to average power
        - 'power' or 'MW': input is already power, will be repeated
    tol : float, default 1e-9
        Tolerance for checking total energy/power consistency

    Returns
    -------
    pd.DataFrame
        1-hourly DataFrame with carriers as index and hourly snapshots as columns
        Units will be MW (power)
    """

    # --- Transpose so snapshots are rows ---
    df_work = df.T.copy()
    df_work.index = pd.to_datetime(df_work.index)
    df_work.index.name = "snapshot"

    # --- Determine time resolution ---
    if len(df_work.index) < 2:
        raise ValueError(
            "DataFrame must have at least 2 time points to determine resolution"
        )

    time_diff = df_work.index[1] - df_work.index[0]
    resolution_hours = time_diff.total_seconds() / 3600

    if resolution_hours <= 0:
        raise ValueError("Time resolution must be positive")

    # logger.info(f"Detected time resolution: {resolution_hours}H")

    # --- Determine unit and conversion factor ---
    if unit.lower() in ["energy", "mwh"]:
        detected_unit = "energy"
        conversion_factor = resolution_hours
        # logger.info(f"Using unit: energy (MWh per {resolution_hours}H block)")
    elif unit.lower() in ["power", "mw"]:
        detected_unit = "power"
        conversion_factor = 1.0
        # logger.info(f"Using unit: power (MW)")
    else:
        raise ValueError("Unit must be 'energy'/'MWh' or 'power'/'MW'")

    # --- Convert to MW if needed ---
    if detected_unit == "energy":
        df_mw = df_work / conversion_factor
    else:
        df_mw = df_work.copy()

    # --- Create 1-hourly index ---
    start = df_mw.index[0]
    # Extend to cover the last block fully
    end = df_mw.index[-1] + pd.Timedelta(hours=resolution_hours - 1)
    hourly_index = pd.date_range(start=start, end=end, freq="1h")

    # --- Reindex and forward-fill ---
    df_mw_reindexed = df_mw.reindex(hourly_index)
    df_mw_1h = df_mw_reindexed.ffill()

    # --- Transpose back so carriers are rows ---
    df_1h = df_mw_1h.T

    # --- Sanity checks ---
    if detected_unit == "energy":
        # For energy: check total energy conservation
        total_original = df.values.sum()
        total_1h = df_1h.values.sum()  # MW values, when multiplied by 1h give MWh

        if not np.isclose(total_original, total_1h, rtol=tol):
            logger.info(
                f"WARNING: total energy mismatch! "
                f"Original sum={total_original:.1f} MWh, "
                f"1H sum={total_1h:.1f} MWh"
            )
    else:
        # For power: check that total energy scales correctly by resolution
        total_original_energy = df.values.sum() * resolution_hours  # MW * hours = MWh
        total_1h_energy = df_1h.values.sum() * 1  # MW * 1h = MWh

        if not np.isclose(total_original_energy, total_1h_energy, rtol=tol):
            logger.info(
                f"WARNING: power scaling mismatch! "
                f"Original total energy={total_original_energy:.1f} MWh, "
                f"1H total energy={total_1h_energy:.1f} MWh"
            )

    return df_1h


def aggregate_by_keywords(df, groups):
    """
    Aggregate rows in df according to keyword groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with row index as technology names.
    groups : dict
        Keys = new aggregated name,
        Values = list of substrings to match in the index.

    Returns
    -------
    pd.DataFrame
    """
    df_out = df.copy()
    for new_name, keywords in groups.items():
        mask = df_out.index.to_series().str.contains("|".join(keywords), case=False)
        if mask.any():
            summed = df_out.loc[mask].sum()
            df_out = df_out.drop(df_out.index[mask])
            df_out.loc[new_name] = summed
    return df_out


def collect_and_summarize_flexibility_provision(
    networks, calc_flexibility_contributions, calc_supply_demand, expand_to_1h, groups
):
    """
    Collect flexibility provision data for all years and save to DataFrame.

    Parameters
    ----------
    networks : dict
        Dictionary with year as key and network object as value
    calc_flexibility_contributions : function
        The flexibility contributions calculation function
    calc_supply_demand : function
        Function to extract supply and demand from network
    expand_to_1h : function
        Function to expand data to hourly resolution

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (Granularity, Year) and technology columns
    """

    all_data = {}

    for year in networks.keys():
        print(f"Processing year {year}...")

        # Extract supply and demand data
        s, d = calc_supply_demand(
            networks[year],
            energy=False,
            interconnectors=True,
            merge_dist_grid=True,
            drop_dist_grid=True,
            add_diff_as_import=True,
        )
        electricity_supply = expand_to_1h(s, unit="power")
        electricity_demand = expand_to_1h(d, unit="power")

        # Calculate flexibility contributions (solutions only)
        flexible_df = calc_flexibility_contributions(
            electricity_supply=electricity_supply,
            electricity_demand=electricity_demand,
            granularity="all",
            analyze="flexible",
        )

        # Extract carrier names (remove Supply_/Demand_ prefix)
        def extract_carrier_name(tech_name):
            if tech_name.startswith("Supply_"):
                return tech_name[7:]
            elif tech_name.startswith("Demand_"):
                return tech_name[7:]
            else:
                return tech_name

        # Reset index and extract carriers
        df_reset = flexible_df.reset_index()
        df_reset["Carrier"] = df_reset["Technology"].apply(extract_carrier_name)

        # Group by granularity and carrier (sum supply and demand)
        df_grouped = df_reset.groupby(["Granularity", "Carrier"])[
            "Contribution (TWh/year)"
        ].sum()

        # Convert to DataFrame with carriers as columns
        df_pivot = df_grouped.unstack("Carrier", fill_value=0)

        # Apply keyword-based aggregation
        for granularity in df_pivot.index:
            gran_data = df_pivot.loc[
                granularity:granularity
            ].T  # Transpose to get carriers as index
            gran_aggregated = aggregate_by_keywords(gran_data, groups)

            # Group small contributions into "Other" (< 1 TWh)
            # Get the first (and only) column since we're working with one granularity at a time
            series_data = gran_aggregated.iloc[:, 0]
            small_mask = series_data.abs() < 1.0

            if small_mask.any():
                other_sum = series_data.loc[small_mask].sum()
                series_data = series_data.drop(series_data.index[small_mask])
                if abs(other_sum) > 0.01:  # Only add "Other" if non-negligible
                    series_data.loc["Other"] = other_sum

            # Store the processed data
            all_data[(granularity, year)] = series_data

    # Combine into single DataFrame
    combined_df = pd.DataFrame(all_data).T
    combined_df.index.names = ["Granularity", "Year"]
    combined_df = combined_df.fillna(0)

    return combined_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        snakemake = mock_snakemake(
            "flexibility_analysis",
            simpl="",
            clusters=27,
            opts="",
            sector_opts="None",
            run="HighFlex",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons
    os.makedirs(os.path.dirname(snakemake.params.output_dir), exist_ok=True)

    # Load networks
    networks = {int(fn[-7:-3]): pypsa.Network(fn) for fn in snakemake.input.networks}

    ####### Calculate flexibility metrics

    # Calculate flexibility needs

    all_flex_needs = {}

    logger.info("Calculating flexibility needs for all years...")
    for year in planning_horizons:
        logger.info(f"Processing year {year}...")

        # Get data for the year
        s, d = calc_supply_demand(
            networks[year],
            energy=False,
        )
        electricity_supply = expand_to_1h(s, unit="power")
        electricity_demand = expand_to_1h(d, unit="power")
        residual_load = calc_residual_load(electricity_supply, electricity_demand)

        # Calculate flexibility needs
        flex_needs = calc_flexibility_needs(residual_load)
        all_flex_needs[year] = flex_needs["Flexibility (TWh/year)"]

    # Create DataFrame with years as columns and granularities as rows
    flex_needs_df = pd.DataFrame(all_flex_needs)

    # Save results
    flex_needs_df.to_csv(snakemake.output.flex_needs)
    logger.info(f"Saved flexibility needs to {snakemake.output.flex_needs}")

    # Calculate flexibility causes and contributions (raw)

    flex_causes_raw = {}
    flex_contributions_raw = {}

    logger.info(
        "Calculating flexibility causes and contributions for all years... (raw)"
    )

    for year in planning_horizons:
        # Extract supply and demand data
        s, d = calc_supply_demand(
            networks[year],
            energy=False,
            interconnectors=False,
            merge_dist_grid=True,
            drop_dist_grid=True,
            add_diff_as_import=True,
        )
        electricity_supply = expand_to_1h(s, unit="power")
        electricity_demand = expand_to_1h(d, unit="power")

        # Calculate flexibility contributions
        flexible_df, inflexible_df = calc_flexibility_contributions(
            electricity_supply=electricity_supply,
            electricity_demand=electricity_demand,
            granularity="all",
            analyze="both",
            print_info=False,
        )

        # aggreate decentral heat pump techs if in flex_causes
        df_clean = inflexible_df.copy()
        hp_groups = {
            "heat pump (decentral)": ["rural air heat pump", "rural ground heat pump", "urban decentral air heat pump"],
        }
        df_agg = pd.concat([
            aggregate_by_keywords(df_clean.xs(gran), hp_groups).assign(Granularity=gran)
            for gran in df_clean.index.get_level_values("Granularity").unique()
        ]).set_index("Granularity", append=True).swaplevel()
        
        flex_causes_raw[year] = df_agg # inflexible_df
        flex_contributions_raw[year] = flexible_df

    # Save raw results as pickle
    with open(snakemake.output.flex_causes_raw, "wb") as f:
        pickle.dump(flex_causes_raw, f)
    logger.info(f"Saved flexibility causes to {snakemake.output.flex_causes_raw}")

    with open(snakemake.output.flex_contributions_raw, "wb") as f:
        pickle.dump(flex_contributions_raw, f)
    logger.info(
        f"Saved flexibility contributions to {snakemake.output.flex_contributions_raw}"
    )

    # Calculate cleaned flexibility contributions
    logger.info("Calculating cleaned flexibility contributions for all years...")

    # Collect and process all flexibility provision data
    flexibility_provision_df = collect_and_summarize_flexibility_provision(
        networks,
        calc_flexibility_contributions,
        calc_supply_demand,
        expand_to_1h,
        tech_groups,
    )

    # Save the data
    flexibility_provision_df.to_csv(snakemake.output.flex_contributions_clean)
    logger.info(
        f"Saved cleaned flexibility contributions to {snakemake.output.flex_contributions_clean}"
    )

    # Calculate flexibility per node
    logger.info("Calculating flexibility needs per node for all years...")

    all_flex_needs_per_node = {}
    bus_supply = {}
    bus_demand = {}
    region = "DE"
    bus_carrier = ["AC"]
    n = networks[planning_horizons[0]]
    buses_de = n.buses[
        (n.buses.index.str[:2] == region) & (n.buses.carrier.isin(bus_carrier))
    ].index

    for year in planning_horizons:
        logger.info(f"Processing year {year}...")

        bus_supply[year] = {}
        bus_demand[year] = {}

        flex_needs_buses = pd.DataFrame(
            index=["daily", "weekly", "annual"], columns=buses_de
        )

        for bus in buses_de:
            s, d = calc_supply_demand(
                networks[year],
                energy=False,
                interconnectors=True,
                merge_dist_grid=True,
                drop_dist_grid=True,
                add_diff_as_import=True,
                buses=[bus, f"{bus} low voltage"],
            )

            electricity_supply = expand_to_1h(s, unit="power")
            electricity_demand = expand_to_1h(d, unit="power")

            bus_supply[year][bus] = electricity_supply
            bus_demand[year][bus] = electricity_demand

            residual_load = calc_residual_load(electricity_supply, electricity_demand)
            flex_needs_buses[bus] = calc_flexibility_needs(residual_load)

        all_flex_needs_per_node[year] = flex_needs_buses

    # Save flexibility needs per node
    with open(snakemake.output.flex_needs_per_node, "wb") as f:
        pickle.dump(all_flex_needs_per_node, f)
    logger.info(
        f"Saved flexibility needs per node to {snakemake.output.flex_needs_per_node}"
    )

    # Flexibility causes and contributions per node

    logger.info(
        "Calculating flexibility causes and contributions per node for all years..."
    )

    flex_contribution_bus = {}
    inflex_cause_bus = {}

    for year in planning_horizons:
        logger.info(f"Calculating flexibility contributions for year {year}...")

        flex_contribution_bus[year] = {}
        inflex_cause_bus[year] = {}

        for bus in buses_de:
            flex_contribution_bus[year][bus], inflex_cause_bus[year][bus] = (
                calc_flexibility_contributions(
                    electricity_supply=bus_supply[year][bus],
                    electricity_demand=bus_demand[year][bus],
                    granularity="all",
                )
            )

    # Save results
    with open(snakemake.output.flex_contributions_per_node, "wb") as f:
        pickle.dump(flex_contribution_bus, f)
    logger.info(
        f"Saved flexibility contributions per node to {snakemake.output.flex_contributions_per_node}"
    )
    with open(snakemake.output.flex_causes_per_node, "wb") as f:
        pickle.dump(inflex_cause_bus, f)
    logger.info(
        f"Saved flexibility causes per node to {snakemake.output.flex_causes_per_node}"
    )

    # MONTHLY ANALYSIS: Calculate flexibility needs
    all_flex_needs_monthly = {}

    logger.info("Calculating flexibility needs (with monthly) for all years...")
    for year in planning_horizons:
        logger.info(f"Processing year {year}...")
        s, d = calc_supply_demand(networks[year], energy=False)
        electricity_supply = expand_to_1h(s, unit="power")
        electricity_demand = expand_to_1h(d, unit="power")
        residual_load = calc_residual_load(electricity_supply, electricity_demand)
        flex_needs = calc_flexibility_needs_with_monthly(residual_load)
        all_flex_needs_monthly[year] = flex_needs["Flexibility (TWh/year)"]

    flex_needs_monthly_df = pd.DataFrame(all_flex_needs_monthly)
    flex_needs_monthly_df.to_csv(snakemake.output.flex_needs_monthly)
    logger.info(f"Saved monthly flexibility needs to {snakemake.output.flex_needs_monthly}")

    # Calculate flexibility causes and contributions (monthly, raw)
    flex_causes_monthly_raw = {}
    flex_contributions_monthly_raw = {}

    logger.info("Calculating flexibility causes and contributions (monthly) for all years...")
    for year in planning_horizons:
        s, d = calc_supply_demand(
            networks[year],
            energy=False,
            interconnectors=False,
            merge_dist_grid=True,
            drop_dist_grid=True,
            add_diff_as_import=True,
        )
        electricity_supply = expand_to_1h(s, unit="power")
        electricity_demand = expand_to_1h(d, unit="power")

        flexible_df, inflexible_df = calc_flexibility_contributions_with_monthly(
            electricity_supply=electricity_supply,
            electricity_demand=electricity_demand,
            granularity="all",
            analyze="both",
            print_info=False,
        )
        flex_causes_monthly_raw[year] = inflexible_df
        flex_contributions_monthly_raw[year] = flexible_df

    with open(snakemake.output.flex_causes_monthly_raw, "wb") as f:
        pickle.dump(flex_causes_monthly_raw, f)
    logger.info(f"Saved monthly flexibility causes to {snakemake.output.flex_causes_monthly_raw}")

    with open(snakemake.output.flex_contributions_monthly_raw, "wb") as f:
        pickle.dump(flex_contributions_monthly_raw, f)
    logger.info(f"Saved monthly flexibility contributions to {snakemake.output.flex_contributions_monthly_raw}")

    # Calculate cleaned monthly flexibility contributions
    logger.info("Calculating cleaned monthly flexibility contributions for all years...")
    flexibility_provision_monthly_df = collect_and_summarize_flexibility_provision(
        networks,
        calc_flexibility_contributions_with_monthly,
        calc_supply_demand,
        expand_to_1h,
        tech_groups,
    )
    flexibility_provision_monthly_df.to_csv(snakemake.output.flex_contributions_monthly_clean)
    logger.info(f"Saved cleaned monthly flexibility contributions to {snakemake.output.flex_contributions_monthly_clean}")
