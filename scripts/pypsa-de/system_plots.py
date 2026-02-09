# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>>
#
# SPDX-License-Identifier: MIT
"""
Plot system characteristics per scenario.

Description
-------

Output plots:
- 
"""
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib import colors, colormaps
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy 
import logging
import os
import sys
from pathlib import Path

import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle

import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from flexibility_analysis import aggregate_by_keywords
from flexibility_utils import (
    df_to_png,
    c1_groups,
    c1_groups_name,
    aggregate_small_contributors,
    find_project_root,
    tech_colors,
    tech_groups,
    year_colors_gradient,
)

logger = logging.getLogger(__name__)

def calculate_storage_capacity(n, scenario, year, region="DE", save_plot=True, plot_dir=None):
    """
    Calculate storage capacity metrics for a single network.
    
    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    scenario : str
        Scenario name (e.g., 'LowFlex', 'MediumFlex')
    year : int
        Year of the scenario
    region : str, optional
        Region code to filter (default: "DE")
    save_plot : bool, optional
        Whether to save the result as PNG (default: True)
    plot_dir : str, optional
        Directory to save plots (required if save_plot=True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 
          capacity metrics by technology
    """
    
    techs = [
        "battery", "home battery", "EV battery", 
        "rural water tanks", "urban central water pits",
        "urban central water tanks", "urban decentral water tanks",
        "H2 Store", "PHS", "Iron-Air battery"
    ]
    
    result = pd.DataFrame(
        index=techs, 
        columns=["energy (GWh)", "discharge (GW)", "charge (GW)"]
    )
    
    kwargs = {
        "groupby": pypsa.statistics.groupers["bus", "carrier"],
        "nice_names": False,
    }
    
    # ============= Storage - Stores =============
    stores_capa = (
        n.stores[n.stores.bus.str.contains(region)]
        [["carrier", "e_nom_opt"]]
        .groupby("carrier")
        .sum() / 1e3
    )
    
    result.loc["battery", "energy (GWh)"] = stores_capa.loc["battery", "e_nom_opt"] 
    result.loc["home battery", "energy (GWh)"] = stores_capa.loc["home battery", "e_nom_opt"] 
    try:
        result.loc["EV battery", "energy (GWh)"] = stores_capa.loc["EV battery", "e_nom_opt"] 
    except KeyError:
        pass
    result.loc["rural water tanks", "energy (GWh)"] = stores_capa.loc["rural water tanks", "e_nom_opt"] 
    result.loc["urban central water pits", "energy (GWh)"] = stores_capa.loc["urban central water pits", "e_nom_opt"] 
    result.loc["urban central water tanks", "energy (GWh)"] = stores_capa.loc["urban central water tanks", "e_nom_opt"] 
    result.loc["urban decentral water tanks", "energy (GWh)"] = stores_capa.loc["urban decentral water tanks", "e_nom_opt"] 
    result.loc["H2 Store", "energy (GWh)"] = stores_capa.loc["H2 Store", "e_nom_opt"]
    try:
        result.loc["Iron-Air battery", "energy (GWh)"] = stores_capa.loc["iron-air battery", "e_nom_opt"]
    except KeyError:
        pass

    # ============= Storage - Storage Units =============
    su_capa = (
        n.storage_units[n.storage_units.bus.str.contains(region)]
        .assign(energy_capacity=lambda x: x["p_nom_opt"] * x["max_hours"] / 1e3)
        .groupby("carrier")["energy_capacity"]
        .sum()
    )
    result.loc["PHS", "energy (GWh)"] = su_capa.loc["PHS"]

    # ============= Elec Components =============
    elec_capas = (
        n.statistics.optimal_capacity(bus_carrier=["AC", "low voltage"], **kwargs)
        .filter(like=region)
        .groupby("carrier")
        .sum()
        .div(1e3)
    )

    result.loc["PHS", "discharge (GW)"] = elec_capas.get("PHS", 0)
    result.loc["PHS", "charge (GW)"] = -elec_capas.get("PHS", 0)

    result.loc["EV battery", "discharge (GW)"] = elec_capas.get("V2G", 0)
    result.loc["EV battery", "charge (GW)"] = elec_capas.get("BEV charger", 0)

    result.loc["Iron-Air battery", "discharge (GW)"] = elec_capas.get("iron-air battery discharger", 0)
    result.loc["Iron-Air battery", "charge (GW)"] = elec_capas.get("iron-air battery charger", 0)

    for tech in ["battery", "home battery"]:
        result.loc[tech, "charge (GW)"] = elec_capas.get(f"{tech} charger", 0)
        result.loc[tech, "discharge (GW)"] = elec_capas.get(f"{tech} discharger", 0)

    # ============= Heat Components =============
    heat_capas = (
        n.statistics.optimal_capacity(
            bus_carrier=["urban central heat", "rural heat", "urban decentral heat"], 
            **kwargs
        )
        .filter(like=region)
        .groupby("carrier")
        .sum()
        .div(1e3)
    )

    for tech in ["rural water tanks",
                "urban central water pits",
                "urban central water tanks",
                "urban decentral water tanks"]:
        result.loc[tech, "charge (GW)"] = heat_capas.get(f"{tech} charger", 0)
        result.loc[tech, "discharge (GW)"] = heat_capas.get(f"{tech} discharger", 0)

    # ============= H2 Components =============
    h2_capas = (
        n.statistics.optimal_capacity(bus_carrier=["H2"], **kwargs)
        .filter(like=region)
        .drop("Store")
        .groupby("carrier")
        .sum()
        .div(1e3)
    )

    result.loc["H2 Store", "discharge (GW)"] = h2_capas.clip(upper=0).sum()
    result.loc["H2 Store", "charge (GW)"] = h2_capas.clip(lower=0).sum()
    
    # ============= Energy-to-Power Ratio =============
    result["energy-to-power (h)"] = (
        result["energy (GWh)"] / result["discharge (GW)"].where(result["discharge (GW)"] != 0, np.nan)
    )

    # ============= Maximum Usage Statistics =============
    kwargs_stats = {
        "groupby": n.statistics.groupers.get_name_bus_and_carrier,
        "nice_names": False,
    }
    buses = n.buses[(n.buses.index.str[:2] == region)].index

    supply = n.statistics.supply(aggregate_time=False, **kwargs_stats)
    demand = n.statistics.withdrawal(aggregate_time=False, **kwargs_stats)

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

    result.loc["PHS", "max discharge (GW)"] = supply.loc["PHS"].max()
    result.loc["PHS", "max charge (GW)"] = demand.loc["PHS"].max()
    try:
        result.loc["EV battery", "max discharge (GW)"] = supply.loc["V2G"].max()
    except KeyError:
        result.loc["EV battery", "max discharge (GW)"] = 0.0
    result.loc["EV battery", "max charge (GW)"] = demand.loc["BEV charger"].max()
    try: 
        result.loc["Iron-Air battery", "max discharge (GW)"] = supply.loc["iron-air battery discharger"].max()
        result.loc["Iron-Air battery", "max charge (GW)"] = demand.loc["iron-air battery charger"].max()
    except KeyError:
        pass

    for tech in ["battery", "home battery"]:
        result.loc[tech, "max charge (GW)"] = demand.loc[f"{tech} charger"].max()
        result.loc[tech, "max discharge (GW)"] = supply.loc[f"{tech} discharger"].max()

    for tech in ["rural water tanks",
                "urban central water pits",
                "urban central water tanks",
                "urban decentral water tanks"]:
        result.loc[tech, "max charge (GW)"] = supply.loc[f"{tech} charger"].max()
        result.loc[tech, "max discharge (GW)"] = demand.loc[f"{tech} discharger"].max()

    result.loc["H2 Store", "max discharge (GW)"] = supply.loc[h2_capas[h2_capas < 0].index].sum().max()
    result.loc["H2 Store", "max charge (GW)"] = supply.loc[h2_capas[h2_capas > 0].index].sum().max()

    result["max charge (GW)"] = result["max charge (GW)"] / 1e3
    result["max discharge (GW)"] = result["max discharge (GW)"] / 1e3

    # ============= Save Plot =============
    if save_plot:
        if plot_dir is None:
            raise ValueError("plot_dir must be specified when save_plot=True")
        df = result.astype(float).round(2)
        df_to_png(df, f'{plot_dir}/storage_capacity_{scenario.lower()}_{year}.png')
    
    return result

def plot_storage_map(
    network,
    technologies,
    onshore_regions,
    stores_capa,
    output_path,
    scenario_name="",
    extent=None,
    figsize=None,
    dpi=300,
    cmap="inferno_r",
    wspace=0.15,
    hspace=0.2,
):
    """
    Plot energy storage capacity maps for different technologies.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with bus location data
    technologies : list
        List of storage carrier names to plot
    onshore_regions : gpd.GeoDataFrame
        Geographic regions for background
    stores_capa : pd.DataFrame
        Storage capacities grouped by ["bus", "carrier"] with "e_nom_opt" column
    output_path : str
        Path to save the figure
    scenario_name : str, optional
        Scenario name for title
    extent : list, optional
        Map extent [lon_min, lon_max, lat_min, lat_max]
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Resolution
    cmap : str, optional
        Colormap name
    wspace : float, optional
        Horizontal spacing between subplots
    hspace : float, optional
        Vertical spacing between subplots
    """
    if extent is None:
        extent = [5.5, 15.5, 47, 56]
    
    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    display_projection = ccrs.EqualEarth()
    
    # Filter available technologies
    available_techs = [t for t in technologies if t in stores_capa.index.get_level_values("carrier").unique()]
    
    if not available_techs:
        raise ValueError(f"No technologies found in stores_capa")
    
    # Calculate layout
    n_plots = len(available_techs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (3.5 * n_cols, 5 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": display_projection},
        figsize=figsize
    )
    
    axes = np.atleast_1d(axes).flatten()
    
    for i, tech in enumerate(available_techs):
        ax = axes[i]
        
        # Follow the working pattern
        df = onshore_regions.copy()
        df = df[df.index.str.contains("DE")]
        
        capas = stores_capa.xs(tech, level="carrier", drop_level=True)["e_nom_opt"]
        capas.index = network.buses.location.loc[capas.index].values
        
        df[tech] = capas
        total_capacity = capas.sum()
        
        vmin, vmax = df[tech].min(), df[tech].max()
        
        # Background
        ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.coastlines(edgecolor="black", linewidth=0.5)
        ax.set_facecolor("white")
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.set_title(f"{tech}\nTotal: {total_capacity:.1f} GWh", fontsize=10, pad=15)
        
        # Plot
        df.to_crs(display_projection.proj4_init).plot(
            column=tech,
            ax=ax,
            linewidth=0.05,
            edgecolor="grey",
            legend=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.set_aspect(aspect_ratio)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, orientation="horizontal")
        cbar.set_label("Energy Capacity (GWh)", fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Energy Storage Capacity - {scenario_name}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return fig

def plot_balance(nb, title="title", tech_colors=None, save_path=None):
    import matplotlib.colors as mcolors
    
    resample = "D"
    nb = nb.resample(resample).mean()
    df = nb
    
    # Check if dataframe is empty or has no numeric data
    if df.empty:
        logger.info(f"WARNING: Skipping plot '{title}' - no data available")
        return
    
    # Check if all values are zero or NaN
    if df.abs().sum().sum() == 0 or df.isna().all().all():
        logger.info(f"WARNING: Skipping plot '{title}' - all values are zero or NaN")
        return
    
    # split into df with positive and negative values
    df_neg, df_pos = df.clip(upper=0), df.clip(lower=0)
    df_pos = df_pos[df_pos.sum().sort_values(ascending=False).index]
    df_neg = df_neg[df_neg.sum().sort_values().index]
    
    # Check if both are empty after splitting
    if df_pos.empty and df_neg.empty:
        logger.info(f"WARNING: Skipping plot '{title}' - no positive or negative values to plot")
        return
    
    # Check if we have any non-zero columns
    df_pos = df_pos.loc[:, (df_pos != 0).any(axis=0)]
    df_neg = df_neg.loc[:, (df_neg != 0).any(axis=0)]
    
    if df_pos.empty and df_neg.empty:
        logger.info(f"WARNING: Skipping plot '{title}' - all columns contain only zeros")
        return
    
    # Helper function to validate and fix colors
    def validate_color(col, color_value):
        if color_value is None:
            logger.info(f"WARNING: Missing color for carrier '{col}', using 'grey'")
            return 'grey'
        try:
            # Try to convert to rgba to validate
            mcolors.to_rgba(color_value)
            return color_value
        except (ValueError, TypeError) as e:
            logger.info(f"WARNING: Invalid color for carrier '{col}': {repr(color_value)} (type: {type(color_value).__name__})")
            logger.info(f"         Error: {e}")
            return 'grey'
    
    # get colors with validation
    c_neg = [
        validate_color(col, tech_colors.get(col, 'grey') if tech_colors else 'grey')
        for col in df_neg.columns
    ]
    c_pos = [
        validate_color(col, tech_colors.get(col, 'grey') if tech_colors else 'grey')
        for col in df_pos.columns
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # plot positive values (only if not empty)
    if not df_pos.empty:
        ax = df_pos.plot.area(ax=ax, stacked=True, color=c_pos, linewidth=0.0)
    
    # rename negative values that are also present on positive side, so that they are not shown and plot negative values
    if not df_neg.empty:
        def f(c):
            return "out_" + c
        
        cols = [f(c) if (c in df_pos.columns) else c for c in df_neg.columns]
        cols_map = dict(zip(df_neg.columns, cols))
        ax = df_neg.rename(columns=cols_map).plot.area(
            ax=ax, stacked=True, color=c_neg, linewidth=0.0
        )
    
    # explicitly filter out duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    
    if not handles:
        logger.info(f"WARNING: Skipping plot '{title}' - no data to display in legend")
        plt.close(fig)
        return
    
    filtered_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if not l.startswith("out_")
    ]
    
    if not filtered_handles_labels:
        logger.info(f"WARNING: Skipping plot '{title}' - no valid legend entries")
        plt.close(fig)
        return
    
    handles, labels = zip(*filtered_handles_labels)
    
    # rescale the y-axis
    y_min = df_neg.sum(axis=1).min() if not df_neg.empty else 0
    y_max = df_pos.sum(axis=1).max() if not df_pos.empty else 0
    
    if y_min == 0 and y_max == 0:
        logger.info(f"WARNING: Skipping plot '{title}' - no variation in data")
        plt.close(fig)
        return
    
    ax.set_ylim([1.05 * y_min, 1.05 * y_max])
    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(1.13, 1.01),
    )
    ax.set_title(title)
    ax.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved plot: {save_path}")
    
    plt.close(fig)

def plot_electricity_capacity(
    networks,
    years,
    c1_groups,
    c1_groups_name,
    tech_colors,
    CAPACITY_DIR,
    *,
    de_only=True,
    scenario_name="scenario",
    ylim=None,
    kwargs=None,
):
    """
    Plot electricity capacity for one scenario.

    Parameters
    ----------
    networks : dict[int, pypsa.Network]
        Dictionary like networks[year]
    years : list[int]
        Years to plot (len must be 2)
    c1_groups, c1_groups_name : list
        Technology aggregation definition
    tech_colors : dict
        Mapping tech -> color
    CAPACITY_DIR : str or Path
        Output directory
    de_only : bool, default True
        If True: only DE capacities (bus_carrier=["AC"], filter "DE")
        If False: whole system
    scenario_name : str
        Used in output filename
    ylim : float or None
        y-axis limit (auto if None)
    kwargs : dict or None
        Passed to statistics.optimal_capacity
    """

    if kwargs is None:
        kwargs = {}

    # -----------------------
    # Capacity extraction
    # -----------------------
    cap_all = pd.DataFrame()

    for year in years:
        n = networks[year]

        if de_only:
            cap = (
                n.statistics.optimal_capacity(bus_carrier=["AC"], **kwargs)
                .filter(like="DE")
                .groupby("carrier")
                .sum()
                .div(1e3)  # MW → GW
                .to_frame(name=year)
            )
        else:
            cap = (
                n.statistics.optimal_capacity(bus_carrier=["AC"], nice_names=False)
                .div(1e3)  # MW → GW
                .droplevel(0)
                .to_frame(name=year)
            )

        cap_all = cap_all.combine_first(cap) if not cap_all.empty else cap

    # Drop non-technologies
    cap_all = cap_all.drop(["load", "load-shedding"], errors="ignore")

    # -----------------------
    # Aggregate technologies
    # -----------------------
    c1_groups = c1_groups.copy()
    c1_groups_name = c1_groups_name.copy()

    if "solar" in c1_groups_name:
        i = c1_groups_name.index("solar")
        del c1_groups[i]
        del c1_groups_name[i]

    df_new = cap_all.copy()
    all_grouped_rows = []

    for group, name in zip(c1_groups, c1_groups_name):
        existing = [d for d in group if d in df_new.index]
        if existing:
            df_new.loc[name] = df_new.loc[existing].sum()
            all_grouped_rows.extend([d for d in existing if d != name])

    cap_all_agg = df_new.drop(index=all_grouped_rows)

    # -----------------------
    # Select top technologies
    # -----------------------
    techs = (
        cap_all_agg.abs()
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(16)
        .index
    )

    df_plot = cap_all_agg.loc[techs].fillna(0)

    sort_year = years[-1]
    sorted_techs = (
        df_plot[sort_year].abs().sort_values(ascending=False).index.tolist()
    )

    # -----------------------
    # Plot
    # -----------------------
    n_years = len(years)

    fig, axes = plt.subplots(
        1,
        n_years,
        figsize=(6 * n_years, 5),
        sharex=False,
        sharey=True,
    )

    # Make axes always iterable
    if n_years == 1:
        axes = [axes]

    for i, year in enumerate(years):
        ax = axes[i]

        year_data = df_plot[year].loc[sorted_techs]

        supply = year_data[year_data >= 0]
        demand = -year_data[year_data < 0]

        pos_supply = np.arange(len(supply))
        pos_demand = np.arange(len(demand)) + len(supply) + 1

        if ylim is not None:
            ax.set_ylim(0, ylim)

        ax.bar(
            pos_supply,
            supply.values,
            color=[tech_colors.get(t, "#1f77b4") for t in supply.index],
            alpha=0.8,
        )
        ax.bar(
            pos_demand,
            demand.values,
            color=[tech_colors.get(t, "#d62728") for t in demand.index],
            alpha=0.8,
        )

        ax.axvline(len(supply) - 0.5, color="black", linestyle="--", linewidth=1.0)

        ax.text(0.02, 0.95, "Supply", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold")
        ax.text(0.98, 0.95, "Demand", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold")

        ymax = max(supply.max() if not supply.empty else 0,
                   demand.max() if not demand.empty else 0)

        for x, y in zip(pos_supply, supply.values):
            ax.text(x, y + 0.01 * ymax, f"{y:.0f}",
                    ha="center", va="bottom", fontsize=6)
        for x, y in zip(pos_demand, demand.values):
            ax.text(x, y + 0.01 * ymax, f"{y:.0f}",
                    ha="center", va="bottom", fontsize=6)

        tech_labels = list(supply.index) + list(demand.index)
        ax.set_xticks(list(pos_supply) + list(pos_demand))
        ax.set_xticklabels(tech_labels, rotation=45, ha="right", fontsize=8)

        ax.set_title(f"{year}", fontsize=16)
        ax.set_ylabel("GW", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()

    scope = "de" if de_only else "all-system"
    plt.savefig(
        f"{CAPACITY_DIR}/electricity-capacity-{scope}-{scenario_name}-{'-'.join(map(str, years))}.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        project_root = find_project_root()
        os.chdir(project_root)

        snakemake = mock_snakemake(
            "system_plots",
            simpl="",
            clusters=27,
            opts="",
            sector_opts="None",
            run="LowBattery",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons
    scenario = scenario=snakemake.wildcards.run

    # Make sure output directory exists
    os.makedirs(snakemake.params.output_dir, exist_ok=True)
    PLOT_DIR = snakemake.params.output_dir

    # Load networks
    networks = {int(fn[-7:-3]): pypsa.Network(fn) for fn in snakemake.input.networks}

    ####### PLOTTING #########
    kwargs = {
        "groupby": pypsa.statistics.groupers["bus", "carrier"],
        "nice_names": False,
    }

    ### Storage Capacity Metrics ###
    for year in planning_horizons:
        n = networks[year]
        result = calculate_storage_capacity(
            n, 
            scenario=scenario, 
            year=year, 
            plot_dir=PLOT_DIR
        )

    ### Capacities Map ###

    base = colormaps.get_cmap("RdPu")
    cmap = colors.LinearSegmentedColormap.from_list("", base(np.linspace(0.15, 1, 256)))
    extent_de = [5.5, 15.5, 47, 56]
    aspect_ratio = (extent_de[1] - extent_de[0]) / (extent_de[3] - extent_de[2])
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    df = onshore_regions.copy()

    for year in planning_horizons:
        n = networks[year]
        stores_capa = n.stores[n.stores.bus.str.contains("DE")][["bus","carrier", "e_nom_opt"]].groupby(["bus" , "carrier"]).sum() / 1e3

        su_capa = (n.storage_units[n.storage_units.bus.str.contains("DE") & (n.storage_units.carrier == "PHS")]
                .assign(e_nom_opt=lambda x: x["p_nom_opt"] * x["max_hours"] / 1e3)
                .groupby(["bus" , "carrier"])["e_nom_opt"].sum())

        stores_capa = pd.concat([stores_capa, su_capa])

        techs = ["PHS", 
                "battery", 
                "home battery", 
                "EV battery",
                "rural water tanks",
                "urban central water pits",
                "urban central water tanks",
                "urban decentral water tanks",
                "H2 Store",
        ]

        if scenario == "HighFlex":  
            techs.append("iron-air battery")

        plot_storage_map(
            n,
            techs,
            onshore_regions,
            stores_capa,
            output_path=f"{PLOT_DIR}/storage_map_{scenario}_{year}.png",
            scenario_name=scenario,
            cmap=cmap,
        )

    ### ENERGYY BALANCES ###

    balances = defaultdict(dict)
    for year in planning_horizons:
        network = networks[year]
        ct = "DE"
        buses = network.buses.index[(network.buses.index.str[:2] == ct)].drop("DE")
        balance = (
            network.statistics.energy_balance(
                aggregate_time=False,
                nice_names=False,
                groupby=["bus", "carrier", "bus_carrier"],
            )
            .loc[:, buses, :, :]
            .droplevel("bus")
        )
        balances[scenario][year] = balance

    carriers_sets = [["AC", "low voltage"],
                     ["urban central heat", "rural heat", "urban decentral heat"],
                     ["H2"],
                     ["co2 stored"],
                     ["oil"],
                     ["renewable oil"],
                     ["gas"],
                     ["renewables gas"],
                     ["solid biomass"],
                     ]

    for year in planning_horizons:

        for carriers in carriers_sets:
            mask = balances[scenario][year].index.get_level_values("bus_carrier").isin(carriers)
            nb = balances[scenario][year][mask].groupby("carrier").sum().div(1e3).T

            
            plot_balance(
                nb,
                title=f"Energy Balance of - '{', '.join(carriers)}' ({scenario}, {year})",
                tech_colors=tech_colors,
                save_path=f"{PLOT_DIR}/energy_balance_{'-'.join(carriers)}_{scenario}_{year}.png",
            )
    ### CAPACITY COMPARISON ###

    # Usage
    # DE only
    plot_electricity_capacity(
        networks=networks,
        years=planning_horizons,
        c1_groups=c1_groups,
        c1_groups_name=c1_groups_name,
        tech_colors=tech_colors,
        CAPACITY_DIR=PLOT_DIR,
        de_only=True,
        scenario_name=scenario,
        ylim=500,
        kwargs=kwargs,
    )

    # Whole system
    plot_electricity_capacity(
        networks=networks,
        years=planning_horizons,
        c1_groups=c1_groups,
        c1_groups_name=c1_groups_name,
        tech_colors=tech_colors,
        CAPACITY_DIR=PLOT_DIR,
        de_only=False,
        scenario_name=scenario,
        ylim=1500,
    )


