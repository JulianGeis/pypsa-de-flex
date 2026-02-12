# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>>
#
# SPDX-License-Identifier: MIT
"""
Plot system characteristics comparing between scenarios.

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
from pypsa.statistics import get_transmission_carriers

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from flexibility_analysis import aggregate_by_keywords
from flexibility_utils import (
    df_to_png,
    scenario_colors,
    sector_colors,
    tech_colors,
    aggregate_small_contributors,
    find_project_root,
    scenario_abbrev,
    tech_colors,
    tech_groups,
)
from export_ariadne_variables import get_export_import

logger = logging.getLogger(__name__)


# Define carrier groups configuration
CARRIER_GROUPS = {
    "electricity": {
        "bus_carrier": ["AC", "low voltage"],
        "drop_stores": True,
        "drop_carriers": ["AC", "DC"],
        "threshold": 1.0,  # TWh
    },
    "heat": {
        "bus_carrier": ["urban decentral heat", "rural heat", "urban central heat"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,
    },
    "H2": {
        "bus_carrier": ["H2"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,
    },
    "oil": {
        "bus_carrier": ["oil"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,
    },
    "gas": {
        "bus_carrier": ["gas"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,
    },
    "co2": {
        "bus_carrier": ["co2"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,  # Mt
    },
    "solid_biomass": {
        "bus_carrier": ["solid biomass"],
        "drop_stores": False,
        "drop_carriers": [],
        "threshold": 1.0,
    },
}

def get_generation_consumption(networks, scenarios, year, carrier_name, config, region="DE", kwargs=None):
    """
    Calculate generation and consumption for a given carrier.
    
    Parameters
    ----------
    networks : dict
        Dictionary of networks by scenario
    scenarios : list
        List of scenario names
    year : int
        Planning horizon year
    carrier_name : str
        Name of the carrier (for output files)
    config : dict
        Configuration with bus_carrier, drop_stores, drop_carriers, threshold
    region : str
        Region filter (default "DE")
    kwargs : dict
        Additional kwargs for statistics
        
    Returns
    -------
    gen_df, con_df : tuple of pd.DataFrame
        Generation and consumption DataFrames
    """
    if kwargs is None:
        kwargs = {
            "groupby": pypsa.statistics.groupers["name", "bus", "carrier"],
            "at_port": True,
            "nice_names": False,
        }
    
    # Generation
    gen_data = {}
    for scenario in scenarios:
        result = (
            networks[scenario][year]
            .statistics.supply(bus_carrier=config["bus_carrier"], **kwargs)
            .filter(like=region)
        )
        
        # Drop Store if configured
        if config["drop_stores"]:
            result = result.drop(["Store"], errors="ignore")
        
        result = (
            result
            .groupby(["carrier"])
            .sum()
            .drop(config["drop_carriers"], errors="ignore")
            .div(1e6)  # Convert to TWh or Mt
        )
        gen_data[scenario] = result
    
    gen_df = pd.concat([gen_data[scenario] for scenario in scenarios], axis=1)
    gen_df.columns = scenarios
    
    # Consumption
    con_data = {}
    for scenario in scenarios:
        result = (
            networks[scenario][year]
            .statistics.withdrawal(bus_carrier=config["bus_carrier"], **kwargs)
            .filter(like=region)
        )
        
        # Drop Store if configured
        if config["drop_stores"]:
            result = result.drop(["Store"], errors="ignore")
        
        result = (
            result
            .groupby(["carrier"])
            .sum()
            .drop(config["drop_carriers"], errors="ignore")
            .div(1e6)  # Convert to TWh or Mt
        )
        con_data[scenario] = result
    
    con_df = pd.concat([con_data[scenario] for scenario in scenarios], axis=1)
    con_df.columns = scenarios
    
    return gen_df, con_df


def plot_storage_timeseries(networks, scenarios, year, carrier, region="DE", output_path=None):
    """Plot storage timeseries for a specific carrier."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario in scenarios:
        n = networks[scenario][year]
        stores_i = n.stores[
            (n.stores.carrier == carrier) & (n.stores.bus.str.contains(region))
        ].index
        
        if len(stores_i) > 0:
            energy_capacity = n.stores.loc[stores_i].e_nom_opt.sum() / 1e6
            (n.stores_t.e[stores_i].sum(axis=1) / 1e6).plot(
                label=f"{scenario} (Capacity: {energy_capacity:.2f} TWh)",
                ax=ax
            )
    
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Energy [TWh]")
    ax.set_title(f"{carrier} Storage - {year}")
    ax.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_marginal_prices(networks, years, compare_scenarios, carriers, plot_dir, ylim=(-50, 400)):
    """
    Plot marginal prices time series and price duration curves for specified carriers.
    
    Parameters
    ----------
    networks : dict
        Nested dictionary with structure networks[scenario][year]
    years : list
        List of years to plot
    compare_scenarios : list
        List of scenarios to compare
    carriers : list
        List of carrier names to include
    plot_dir : str
        Directory path to save plots
    ylim : tuple, optional
        Y-axis limits as (min, max), default is (-50, 400)
    """
    for year in years:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        
        for scenario in compare_scenarios:
            buses = networks[scenario][year].buses.index[
                (networks[scenario][year].buses.index.str[:2] == "DE")
                & (networks[scenario][year].buses.carrier.isin(carriers))
            ]
            prices = networks[scenario][year].buses_t.marginal_price[buses]
            
            # plot prices
            ax.plot(prices.mean(axis=1), label=f"{scenario} - {year}", ls=":")
            
            # plot pdc
            sorted_prices = np.sort(prices.mean(axis=1).values)
            pdc = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
            ax2.plot(pdc, sorted_prices, label=f"{scenario} - {year}", ls=":")
        
        # Configure time series plot
        ax.set_title(f"Marginal Prices in {year} ({', '.join(carriers)})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid()
        ax.set_ylim(ylim)
        fig.savefig(
            f"{plot_dir}/marginal_prices_{'_'.join(compare_scenarios)}_{year}_{'_'.join(carriers)}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2
        )
        
        # Configure price duration curve plot
        ax2.set_title(f"Price Duration Curve in {year} ({', '.join(carriers)})")
        ax2.set_xlabel("PDC")
        ax2.set_ylabel("Price (EUR/MWh)")
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid()
        ax2.set_ylim(ylim)
        fig2.savefig(
            f"{plot_dir}/marginal_prices_pdc_{'_'.join(compare_scenarios)}_{year}_{'_'.join(carriers)}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2
        )
        
        plt.close(fig)
        plt.close(fig2)

# calc system cost
def get_tsc(n, country, region="DE"):
    pypsa.options.set_option("params.statistics.drop_zero", False)
    capex = n.statistics.capex(
        groupby=pypsa.statistics.groupers["name", "carrier"], nice_names=False
    )

    opex = n.statistics.opex(
        groupby=pypsa.statistics.groupers["name", "carrier"], nice_names=False
    )

    # filter inter country transmission lines and links
    inter_country_lines = n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(
        n.buses.country
    )
    inter_country_links = n.links.bus0.map(n.buses.country) != n.links.bus1.map(
        n.buses.country
    )
    #
    transmission_carriers = get_transmission_carriers(n).get_level_values("carrier")
    transmission_lines = (
        n.lines.carrier.isin(transmission_carriers) & n.lines.active
    )
    transmission_links = (
        n.links.carrier.isin(transmission_carriers) & n.links.active
    )
    #
    country_transmission_lines = (
        (n.lines.bus0.str.contains(country)) & ~(n.lines.bus1.str.contains(country))
    ) | (
        ~(n.lines.bus0.str.contains(country)) & (n.lines.bus1.str.contains(country))
    )
    country_tranmission_links = (
        (n.links.bus0.str.contains(country)) & ~(n.links.bus1.str.contains(country))
    ) | (
        ~(n.links.bus0.str.contains(country)) & (n.links.bus1.str.contains(country))
    )
    #
    inter_country_transmission_lines = (
        inter_country_lines & transmission_lines & country_transmission_lines
    )
    inter_country_transmission_links = (
        inter_country_links & transmission_links & country_tranmission_links
    )
    inter_country_transmission_lines_i = inter_country_transmission_lines[
        inter_country_transmission_lines
    ].index
    inter_country_transmission_links_i = inter_country_transmission_links[
        inter_country_transmission_links
    ].index
    inter_country_transmission_i = inter_country_transmission_lines_i.union(
        inter_country_transmission_links_i
    )

    #
    tsc = pd.concat([capex, opex], axis=1, keys=["capex", "opex"])
    tsc = tsc.reset_index().set_index("name")
    tsc.loc[inter_country_transmission_i, ["capex", "opex"]] = (
        tsc.loc[inter_country_transmission_i, ["capex", "opex"]] / 2
    )
    tsc.rename(
        index={
            index: index + " " + country for index in inter_country_transmission_i
        },
        inplace=True,
    )
    # rename inter region links and lines
    to_rename_links = n.links[
        (n.links.bus0.str.contains(region))
        & (n.links.bus1.str.contains(region))
        & ~(n.links.index.str.contains(region))
    ].index
    to_rename_lines = n.lines[
        (n.lines.bus0.str.contains(region))
        & (n.lines.bus1.str.contains(region))
        & ~(n.lines.index.str.contains(region))
    ].index
    tsc.rename(
        index={index: index + " " + region for index in to_rename_links},
        inplace=True,
    )
    tsc.rename(
        index={index: index + " " + region for index in to_rename_lines},
        inplace=True,
    )

    tsc = tsc.filter(like=country, axis=0)
    
    tsc_sum = (
        tsc.filter(like=country, axis=0)
        .drop("component", axis=1)
        .groupby("carrier")
        .sum()
    )

    return tsc_sum, tsc


def get_trade_cost(n, region, carriers):
    """
    Positive values mean cost for the domestic energy system (imports > exports)
    Negative values mean revenue for the domestic energy system (exports > imports)
    """
    export_revenue, import_cost = get_export_import(n, region, carriers, unit="€")
    return import_cost - export_revenue

# Define sector mapping
bus_carrier_to_sector = {
    "Electricity": [
        "AC", 
        "low voltage",
        "battery",
        "iron-air battery", 
        "EV battery",
        "home battery",
        "DSM"
    ],
    "Heat": [
        "urban central heat",
        "urban central water tanks",
        "urban central water pits",
        "rural heat",
        "rural water tanks",
        "urban decentral heat",
        "urban decentral water tanks"
    ],
    "H2": [
        "H2"
    ],
    "Gas": [
        "gas",
        "gas primary",
        "biogas",
        "gas for industry",
        "renewable gas"
    ],
    "Coal": [
        "lignite",
        "coal",
        "coal for industry"
    ],
    "Fuels": [
        "oil",
        "oil primary",
        "land transport oil",
        "shipping oil",
        "kerosene for aviation",
        "agriculture machinery oil",
        "renewable oil",
        "naphtha for industry",
        "methanol",
        "industry methanol",
        "shipping methanol",
    ],
    "Biomass": [
        "solid biomass",
        "solid biomass for industry"
    ],
    "CO2": [
        "co2",
        "co2 stored",
        "co2 sequestered",
        "process emissions"
    ]
}


# Function to get bus for each component
def get_component_bus(row, n):
    """Get the bus for a component based on its type."""
    name = row.name
    component = row['component']
    
    if component == 'Generator':
        return n.generators.loc[name, 'bus'] if name in n.generators.index else None
    elif component == 'Store':
        return n.stores.loc[name, 'bus'] if name in n.stores.index else None
    elif component == 'StorageUnit':
        return n.storage_units.loc[name, 'bus'] if name in n.storage_units.index else None
    elif component == 'Link':
        # For links, use bus1 (output bus) - adjust if needed
        return n.links.loc[name, 'bus1'] if name in n.links.index else n.links[n.links.carrier == row.carrier].bus1.head(1).values[0]
    elif component == 'Load':
        return n.loads.loc[name, 'bus'] if name in n.loads.index else None
    elif component == 'Line':
            return "DE0 0"
    else:
        return None

# Map bus to sector
def map_bus_to_sector(bus_name, n, bus_carrier_to_sector):
    """Map a bus to its sector based on carrier."""
    if pd.isna(bus_name):
        return "Other"
    
    # Get bus carrier
    if bus_name in n.buses.index:
        bus_carrier = n.buses.loc[bus_name, 'carrier']
    else:
        return "Other"
    
    # Find matching sector
    for sector, carriers in bus_carrier_to_sector.items():
        if bus_carrier in carriers:
            return sector
    
    return "Other"


def plot_price_duration_curves(
    networks,
    year,
    scenarios,
    carriers=["AC", "low voltage"],
    regions=["DE"],
    scenario_colors=scenario_colors,
    output_dir=None,
    ylim=(-50, 500),
    table_smaller=1,
    table_bigger=300,
    close_plot=True,
):
    """Plot electricity price duration curves comparison across scenarios and years."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scenario_data = {}
    
    # ---- Process each scenario ----
    for scenario in scenarios:
        n = networks[scenario][year]
        buses = (
            n.buses[
                n.buses.carrier.isin(carriers)
                & n.buses.index.str.startswith(tuple(regions))
            ].index
        )
        lmps = n.buses_t.marginal_price[buses].values.flatten()
        lmps_sorted = np.sort(lmps)[::-1]
        pct = np.arange(len(lmps_sorted)) / len(lmps_sorted) * 100
        ax.plot(
            pct,
            lmps_sorted,
            label=scenario,
            color = scenario_colors.get(scenario, 'grey'),
            linewidth=2,
        )
        # metrics
        scenario_data[scenario] = {
            "avg": lmps_sorted.mean(),
            "std": lmps_sorted.std(),
            f"<{table_smaller}": (lmps_sorted < table_smaller).mean() * 100,
            f">{table_bigger}": (lmps_sorted > table_bigger).mean() * 100,
        }
    
    # Calculate dynamic position for quantile labels based on ylim
    y_range = ylim[1] - ylim[0]
    label_y_position = ylim[0] + 0.02 * y_range  # 5% above the bottom
    
    # ---- Vertical quantile lines (x in %) ----
    quantiles = [0, 10, 25, 50, 75, 100]
    for q in quantiles:
        ax.axvline(q, ls="--", lw=0.8, color="grey", alpha=0.5)
        ax.text(q + 0.5, label_y_position, f"{q}%", rotation=90, fontsize=8, alpha=0.7)
    
    ax.set(
        ylim=ylim,
        xlim=(-5, 105),  # Added negative space on x-axis
        xlabel="Percentage of time",
        ylabel="€/MWh",
        # title=f"Price duration curves {year}",
    )
    ax.grid(alpha=0.3, linestyle=':')
    
    # ---- Create fancy compact table ----
    table_data = []
    for sc in scenarios:
        m = scenario_data[sc]
        table_data.append([
            sc,
            f"{m['avg']:.2f}",
            f"{m['std']:.2f}",
            f"{m[f'<{table_smaller}']:.0f}%",
            f"{m[f'>{table_bigger}']:.0f}%",
        ])
    
    col_labels = ["Scenario", "Avg", "Std", f"<{table_smaller}", f">{table_bigger}"]
    table = plt.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="upper right",
        cellLoc="center",
        colColours=["#34495e"] * len(col_labels),
        fontsize=7,
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(0.5, 1.2)  # Reduce narrower columns
    
    # Color the scenario cells
    for i, sc in enumerate(scenarios, start=1):
        table[i, 0].set_facecolor(scenario_colors.get(sc, 'grey'))
        table[i, 0].set_text_props(weight='bold', color='white')
    
    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_text_props(weight='bold', color='white', fontsize=6.5)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir, bbox_inches="tight", dpi=300)
    if close_plot:
        plt.close()
    else:
        plt.show()


def bar_plot_variables(variables, 
                         scenarios, 
                         years,
                         tech_colors, 
                         plot_vars=None,
                         sign_flip_vars=["Electricity", "Hydrogen", "eFuels"],
                         title=None, 
                         ylabel="TWh/a",
                         output_dir=None):
    """Plot multiple trade variables across scenarios for each year in one plot."""
    
    
    # Define the default variables to plot
    if plot_vars is None:
        plot_vars = {
            "Electricity": "Trade|Secondary Energy|Electricity|Volume",
            "Gas": "Primary Energy|Gas",
            "Oil": "Primary Energy|Oil",
            "Hydrogen": "Trade|Secondary Energy|Hydrogen|Volume", 
            "eFuels": "Trade|Secondary Energy|Efuels|Volume",
            "Biomass": "Trade|Primary Energy|Biomass|Net Imports",
        }
    
    for year in years:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Prepare data
        data = {var_name: [] for var_name in plot_vars.keys()}
        
        for scenario in scenarios:
            df = variables[scenario]
            for var_name, var_path in plot_vars.items():
                try:
                    # Extract scalar value properly
                    value = df.loc[var_path, year]
                    # Convert to scalar if it's a Series
                    if hasattr(value, 'values'):
                        value = value.values[0]
                    # Convert to float to ensure it's a scalar
                    value = float(value)
                    # Multiply by -1 for Electricity and Hydrogen
                    if var_name in sign_flip_vars:
                        value = value * -1
                except (KeyError, IndexError):
                    value = 0.0
                
                data[var_name].append(value)
        
        # Set up bar positions
        x = np.arange(len(scenarios))
        n_vars = len(plot_vars)
        width = 0.15  # Width of each bar
        
        # Calculate offset to center the group of bars
        total_width = width * n_vars
        start_offset = -total_width / 2 + width / 2
        
        # Plot bars for each variable
        for i, var_name in enumerate(plot_vars.keys()):
            offset = start_offset + width * i
            color = tech_colors.get(var_name if var_name != "Hydrogen" else "H2", "gray")
            bars = ax.bar(x + offset, data[var_name], width, 
                          label=var_name,
                          color=color,
                          edgecolor='black', 
                          linewidth=1.2,
                          alpha=0.85)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.5:  # Only show labels for non-negligible values
                    if height >= 0:
                        va = 'bottom'
                        y_pos = height
                    else:
                        va = 'top'
                        y_pos = height
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                           f'{height:.1f}',
                           ha='center', va=va, fontsize=8, fontweight='bold')
        
        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1.5)
        
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir + f"{year}.png", bbox_inches="tight", dpi=300)


def plot_curtailment(networks, scenarios, year, tech_colors, output_dir=None):
    """Plot curtailment for wind and solar technologies across scenarios."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define technology groupings
    wind_techs = ['onwind', 'offwind-ac', 'offwind-dc']
    solar_techs = ['solar', 'solar rooftop', 'solar-hsat']
    
    # Prepare data structure
    scenario_data = {sc: {'wind': {}, 'solar': {}} for sc in scenarios}
    
    for scenario in scenarios:
        n = networks[scenario][year]
        
        # Calculate curtailment
        curtailment = (
            n.statistics.curtailment(bus_carrier=["AC", "low voltage"], **kwargs)
            .filter(like="DE")
            .groupby("carrier")
            .sum()
        )
        
        # Extract wind data
        for tech in wind_techs:
            if tech in curtailment.index:
                scenario_data[scenario]['wind'][tech] = curtailment[tech] / 1e6  # Convert to TWh
            else:
                scenario_data[scenario]['wind'][tech] = 0.0
        
        # Extract solar data
        for tech in solar_techs:
            if tech in curtailment.index:
                scenario_data[scenario]['solar'][tech] = curtailment[tech] / 1e6  # Convert to TWh
            else:
                scenario_data[scenario]['solar'][tech] = 0.0
    
    # Set up bar positions
    x = np.arange(len(scenarios))
    width = 0.35

    # Plot wind bars (left)
    bottom_wind = np.zeros(len(scenarios))
    for tech in wind_techs:
        values = [scenario_data[sc]['wind'][tech] for sc in scenarios]
        ax.bar(x - width/2, values, width, bottom=bottom_wind,
               label=tech, color=tech_colors[tech], edgecolor='black', linewidth=0.5)
        bottom_wind += values
    
    # Plot solar bars (right)
    bottom_solar = np.zeros(len(scenarios))
    for tech in solar_techs:
        values = [scenario_data[sc]['solar'][tech] for sc in scenarios]
        ax.bar(x + width/2, values, width, bottom=bottom_solar,
               label=tech, color=tech_colors[tech], edgecolor='black', linewidth=0.5)
        bottom_solar += values
    
    # Calculate max value for y-axis limit
    max_value = max(max(bottom_wind), max(bottom_solar))
    
    # Add only total sum labels on top of bars
    for i, scenario in enumerate(scenarios):
        wind_total = sum(scenario_data[scenario]['wind'].values())
        solar_total = sum(scenario_data[scenario]['solar'].values())
        
        # Sum on top of bars
        if wind_total > 0:
            ax.text(x[i] - width/2, wind_total + 1, f'{wind_total:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if solar_total > 0:
            ax.text(x[i] + width/2, solar_total + 1, f'{solar_total:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax.set_ylabel("TWh/a", fontsize=12)
    # ax.set_title(f"Renewable Curtailment - {year}", fontsize=13, fontweight='bold')
    
    # Set y-axis limit with extra space at top (15% more than max value)
    ax.set_ylim(0, max_value * 1.15)
    
    # Create legend in upper right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f"curtailment_{year}.png", bbox_inches="tight", dpi=300)


def plot_energy_system_cost_comparison(
    df, 
    scenarios=['LowFlex', 'LowBattery', 'Base', 'HighFlex'],
    sectors=['Electricity', 'Heat', 'H2', 'Fuels', 'Gas', 'Biomass', 'Other'],
    colors=None,
    save_path=None,
    figsize=(8, 5)
):
    """
    Plot total energy system cost comparison across scenarios.
    
    Left bars show costs stacked by type (CAPEX, OPEX, Trade).
    Right bars show costs stacked by sector within each cost type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cost data indexed by cost categories (e.g., 'CAPEX Electricity')
        and columns for each scenario
    scenarios : list
        List of scenario names (columns in df)
    sectors : list
        List of sector names to extract from the index
    tech_colors : dict, optional
        Color mapping for technologies/sectors. If None, uses default colors.
    year : int or str, optional
        Year to include in the saved filename
    save_path : str, optional
        Full path to save the figure. If None, doesn't save.
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    
    # Extract data by cost type (left bar)
    capex = np.array([df.filter(like='CAPEX', axis=0)[scenario].sum() for scenario in scenarios])
    opex = np.array([df.filter(like='OPEX', axis=0)[scenario].sum() for scenario in scenarios])
    trade = np.array([df.filter(like='Trade', axis=0)[scenario].sum() for scenario in scenarios])
    total = capex + opex + trade
    
    # Extract data by sector AND cost type (right bar)
    sector_capex = np.zeros((len(scenarios), len(sectors)))
    sector_opex = np.zeros((len(scenarios), len(sectors)))
    sector_trade = np.zeros((len(scenarios), len(sectors)))
    
    for i, scenario in enumerate(scenarios):
        for j, sector in enumerate(sectors):
            sector_capex[i, j] = df[df.index.str.contains(f'CAPEX.*{sector}')][scenario].sum()
            sector_opex[i, j] = df[df.index.str.contains(f'OPEX.*{sector}')][scenario].sum()
            sector_trade[i, j] = df[df.index.str.contains(f'Trade.*{sector}')][scenario].sum()
    
    # Calculate % difference from Base scenario
    base_idx = scenarios.index('Base') if 'Base' in scenarios else None
    if base_idx is not None:
        base_total = total[base_idx]
        pct_diff = (total - base_total) / base_total * 100
    else:
        pct_diff = np.zeros(len(scenarios))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Left bar: stacked by cost type
    ax.bar(x - width/2, capex, width, color='thistle')
    ax.bar(x - width/2, opex, width, bottom=capex, color='pink')
    ax.bar(x - width/2, trade, width, bottom=capex+opex, color='navajowhite')
    
    # Add horizontal text labels on left bars
    for i in range(len(scenarios)):
        ax.text(i - width/2, capex[i]/2, 'CAPEX', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        ax.text(i - width/2, capex[i] + opex[i]/2, 'OPEX', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(i - width/2, capex[i] + opex[i] + trade[i]/2, 'Trade', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Add total cost labels on top of left bars
    for i in range(len(scenarios)):
        ax.text(i - width/2, total[i], f'{total[i]:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Right bar: stack CAPEX by sector, then OPEX by sector, then Trade by sector
    bottom = np.zeros(len(scenarios))
    
    # First layer: CAPEX by sector
    for j, sector in enumerate(sectors):
        ax.bar(x + width/2, sector_capex[:, j], width, bottom=bottom, 
               color=colors[sector], label=sector, edgecolor='white', linewidth=0.5)
        bottom += sector_capex[:, j]
    
    capex_top = bottom.copy()
    
    # Second layer: OPEX by sector
    for j, sector in enumerate(sectors):
        ax.bar(x + width/2, sector_opex[:, j], width, bottom=bottom, 
               color=colors[sector], edgecolor='white', linewidth=0.5)
        bottom += sector_opex[:, j]
    
    opex_top = bottom.copy()
    
    # Third layer: Trade by sector
    for j, sector in enumerate(sectors):
        ax.bar(x + width/2, sector_trade[:, j], width, bottom=bottom, 
               color=colors[sector], edgecolor='white', linewidth=0.5)
        bottom += sector_trade[:, j]
    
    # Add thick horizontal lines between sections
    for i in range(len(scenarios)):
        ax.hlines(capex_top[i], i + width/2 - width/2, i + width/2 + width/2, 
                  colors='black', linewidth=2.5, zorder=10)
        ax.hlines(opex_top[i], i + width/2 - width/2, i + width/2 + width/2, 
                  colors='black', linewidth=2.5, zorder=10)
    
    # Add percentage labels (skip Base scenario)
    if base_idx is not None:
        for i, scenario in enumerate(scenarios):
            if scenario != 'Base':
                sign = '+' if pct_diff[i] > 0 else ''
                ax.text(i + width/2, total[i], f'{sign}{pct_diff[i]:.1f}%', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Cost (billion €)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=9, ncol=4, loc='upper right', title='Sectors')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(total) * 1.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        project_root = find_project_root()
        os.chdir(project_root)

        snakemake = mock_snakemake(
            "system_plots_scenario_comparison",
        )

    configure_logging(snakemake)
    config = snakemake.config

    # Get configuration
    scenarios = snakemake.params.scenarios_to_compare
    planning_horizons = snakemake.params.planning_horizons

    logger.info(f"Loading data for scenarios: {scenarios}")
    logger.info(f"Planning horizons: {planning_horizons}")

    # Initialize dictionaries
    networks = {}

    # Load networks
    logger.info("Loading networks...")
    for network_path in snakemake.input.networks:
        path_parts = Path(network_path).parts
        scenario = path_parts[-3]
        filename = Path(network_path).stem
        year = int(filename.split("_")[-1])

        if scenario not in networks:
            networks[scenario] = {}

        if Path(network_path).exists():
            logger.info(f"  Loading {scenario} network for year {year}...")
            networks[scenario][year] = pypsa.Network(network_path)

    # Load ariadne variables
    variables = {}
    for file in snakemake.input.exported_variables:
        _df = pd.read_excel(
            file, index_col=list(range(5)), sheet_name="data"
        ).droplevel(["Model", "Region"])
        variables[_df.index.get_level_values("Scenario").unique()[0]] = _df.droplevel("Scenario", axis=0)   


    # Create output directory
    output_dir = Path(snakemake.params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("All data loaded successfully!")
    logger.info(f"Loaded {len(networks)} scenarios with networks")

    kwargs = {
    "groupby": pypsa.statistics.groupers["bus", "carrier"],
    "nice_names": False,
    }

    ####### PLOTTING #########

    ## Capacities ##

    carrier_sets = [["AC", "low voltage"],
                    ["urban central heat", "rural heat", "urban decentral heat"],
                    ["H2"],
                    ["oil"],
                    ["gas"],
                    ["co2 stored"],
                    ["co2 sequestered"],
                    ]
    region = "DE"
    xlims = {
        2020: (-100, 100),
        2025: (-100, 100),
        2030: (-200, 200),
        2035: (-300, 300),
        2040: (-400, 400),
        2045: (-600, 700),
    }

    for i, year in enumerate(planning_horizons):
        for bc in carrier_sets: 
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 15))  
                capas = {} 
                
                for scenario in scenarios:
                    capas[scenario] = (
                        networks[scenario][year]
                        .statistics.optimal_capacity(bus_carrier=bc, **kwargs)
                        .filter(like=region)
                        .groupby("carrier")
                        .sum()
                        .div(1e3)
                    )
                
                df = pd.concat([capas[scenario] for scenario in scenarios], axis=1)
                df.columns = scenarios
                
                # Skip if dataframe is empty
                if df.empty:
                    logger.info(f"Skipping {bc} for {year}: No data")
                    plt.close(fig)
                    continue
                
                df.plot(
                    kind="barh",
                    title=f"Optimal capacity in {year} ({bc})",
                    xlim=xlims[year],
                    ylabel="GW",
                    ax=ax,
                )
                plt.grid()
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/capacity_{'_'.join(scenarios)}_{bc}_{year}.png", 
                    dpi=300, 
                    bbox_inches="tight"
                )
                plt.close(fig)
                
            except (IndexError, KeyError) as e:
                logger.info(f"Skipping {bc} for {year}: {e}")
                plt.close(fig)
                continue

    # CAPACITIES TABLES
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
        for bc in carriers_sets:
            capa = {}
            for scenario in scenarios:
                stats = (
                    networks[scenario][year]
                    .statistics.optimal_capacity(bus_carrier=bc, **kwargs)
                    .filter(like="DE")
                )
                if stats.empty:
                    capa[scenario] = pd.Series(dtype=float)
                else:
                    capa[scenario] = stats.groupby(["carrier"]).sum().div(1e3)  # GW
            
            capa_df = pd.concat([capa[scenario] for scenario in scenarios], axis=1)
            capa_df.columns = scenarios
            
            if not capa_df.empty:
                df = round(capa_df[capa_df.gt(1).any(axis=1)], 2)
                if not df.empty:
                    df_to_png(df, f"{output_dir}/capacity_{'-'.join(bc)}_{'-'.join(scenarios)}_{year}.png")

    ### GENERATION & CONSUMPTION ###

    # Setup kwargs for statistics
    kwargs = {
        "groupby": pypsa.statistics.groupers["name", "bus", "carrier"],
        "at_port": True,
        "nice_names": False,
    }
    
    for year in planning_horizons:
        logger.info(f"Analyzing year {year} for scenarios: {scenarios}")
        
        # ==================== PROCESS ALL CARRIERS ====================
        for i, (carrier_name, config) in enumerate(CARRIER_GROUPS.items(), 1):
            
            # Get generation and consumption
            gen_df, con_df = get_generation_consumption(
                networks, scenarios, year, carrier_name, config, region, kwargs
            )
            
            # Save generation
            if not gen_df.empty and gen_df.gt(config["threshold"]).any().any():
                df = round(gen_df[gen_df.gt(config["threshold"]).any(axis=1)], 2)
                df_to_png(df, os.path.join(output_dir, f'{carrier_name}_generation_{"_".join(scenarios)}_{year}.png'))
            
            # Save consumption
            if not con_df.empty and con_df.gt(config["threshold"]).any().any():
                df = round(con_df[con_df.gt(config["threshold"]).any(axis=1)], 2)
                df_to_png(df, os.path.join(output_dir, f'{carrier_name}_consumption_{"_".join(scenarios)}_{year}.png'))
        
        # ==================== HEAT BY LEVEL ====================
        logger.info("\nProcessing heat by level...")
        heat_levels = CARRIER_GROUPS["heat"]["bus_carrier"]
        for bc in heat_levels:
            heat_gen = {}
            for scenario in scenarios:
                heat_gen[scenario] = (
                    networks[scenario][year]
                    .statistics.supply(bus_carrier=[bc], **kwargs)
                    .filter(like=region)
                    .groupby(["carrier"])
                    .sum()
                    .div(1e6)
                )
            heat_gen_df = pd.concat([heat_gen[scenario] for scenario in scenarios], axis=1)
            heat_gen_df.columns = scenarios
            
            if not heat_gen_df.empty and heat_gen_df.gt(0.1).any().any():
                df = round(heat_gen_df[heat_gen_df.gt(0.1).any(axis=1)], 2)
                filename = bc.replace(" ", "_")
                df_to_png(df, os.path.join(output_dir, f'heat_gen_{filename}_{year}.png'))
        
        # ==================== STORAGE PLOTS ====================
        logger.info("\nProcessing storage plots...")
        
        # Heat storage
        heat_stores_carriers = [
            "urban central water tanks",
            "urban central water pits",
            "urban decentral water tanks",
            "urban decentral water pits",
            "rural water tanks",
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for scenario in scenarios:
            n = networks[scenario][year]
            stores_i = n.stores[
                (n.stores.carrier.isin(heat_stores_carriers)) & (n.stores.bus.str.contains(region))
            ].index
            
            if len(stores_i) > 0:
                energy_capa = n.stores.loc[stores_i].e_nom_opt.sum() / 1e6
                (n.stores_t.e[stores_i].sum(axis=1) / 1e6).plot(
                    label=f"{scenario} (Capacity: {energy_capa:.2f} TWh)",
                    ax=ax
                )
        
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Energy [TWh]")
        ax.set_title(f"Heat Storage - {year}")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'heat_storage_{"_".join(scenarios)}_{year}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # H2 storage
        plot_storage_timeseries(
            networks, scenarios, year, "H2 Store", region,
            os.path.join(output_dir, f'h2_storage_{"_".join(scenarios)}_{year}.png')
        )
        
        # Oil storage
        plot_storage_timeseries(
            networks, scenarios, year, "oil", region,
            os.path.join(output_dir, f'oil_storage_{"_".join(scenarios)}_{year}.png')
        )
        
        # Gas storage
        plot_storage_timeseries(
            networks, scenarios, year, "gas", region,
            os.path.join(output_dir, f'gas_storage_{"_".join(scenarios)}_{year}.png')
        )
        
        # ==================== RENEWABLE OIL/GAS ====================
        logger.info("\nProcessing renewable oil/gas...")
        
        # Renewable oil
        re_oil_config = {"bus_carrier": ["renewable oil"], "drop_stores": False, "drop_carriers": [], "threshold": 1.0}
        re_oil_gen_df, _ = get_generation_consumption(
            networks, scenarios, year, "renewable_oil", re_oil_config, region, kwargs
        )
        
        if not re_oil_gen_df.empty and re_oil_gen_df.gt(1.0).any().any():
            df = round(re_oil_gen_df[re_oil_gen_df.gt(1.0).any(axis=1)], 2)
            df_to_png(df, os.path.join(output_dir, f're_oil_generation_{"_".join(scenarios)}_{year}.png'))
        
        # Renewable gas
        re_gas_config = {"bus_carrier": ["renewable gas"], "drop_stores": False, "drop_carriers": [], "threshold": 1.0}
        re_gas_gen_df, _ = get_generation_consumption(
            networks, scenarios, year, "renewable_gas", re_gas_config, region, kwargs
        )
        
        if not re_gas_gen_df.empty and re_gas_gen_df.gt(1.0).any().any():
            df = round(re_gas_gen_df[re_gas_gen_df.gt(1.0).any(axis=1)], 2)
            df_to_png(df, os.path.join(output_dir, f're_gas_generation_{"_".join(scenarios)}_{year}.png'))
        
    logger.info(f"\n✓ All generation and consumption plots saved to: {output_dir}")


    ### PRICES ###
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
        logger.info(f"\nAnalyzing marginal prices for year {year}...")
        for carriers in carriers_sets:
            try:
                plot_marginal_prices(networks, [year], scenarios, carriers, output_dir)
            except Exception as e:
                logger.warning(f"✗ Failed to plot marginal prices for carriers: {carriers} - {e}")

    ### PRICES - PAPER ###

    price_carriers_to_plot = [
        ["AC", "low voltage"],
        ["urban central heat"],
        ["urban decentral heat"],
        ["rural heat"],
        ["urban central heat", "urban decentral heat", "rural heat"],
        ["H2"],
        ["gas"],
        ["renewable gas"],
        ["methanol"],
        ["renewable oil"],
        ["oil"],
    ]

    for year in planning_horizons:
        for c in price_carriers_to_plot:
            plot_price_duration_curves(
                networks,
                year,
                scenarios,
                carriers=c,
                regions=["DE"],
                scenario_colors=scenario_colors,
                output_dir=output_dir / f"pdc_{'_'.join(c)}_{year}.png",
                ylim=(-50, 500),
                table_smaller=1,
                table_bigger=200,
            )
        
    ### SYSTEM COSTS ###
    # Plot system cost per year

    for year in planning_horizons:

        region = "DE"
        var_all = pd.DataFrame(columns=scenarios)
        tsc_all = {}

        for scenario in scenarios:
            n = networks[(scenario)][year]
            var = pd.Series()

            var["Total Energy System Cost|Trade|Electricity"] = (
                get_trade_cost(n, region, ["AC"]) + get_trade_cost(n, region, ["DC"])
            ) / 1e9
            var["Total Energy System Cost|Trade|eFuels"] = (
                get_trade_cost(n, region, ["renewable oil", "renewable gas", "methanol"]) / 1e9
            )
            var["Total Energy System Cost|Trade|H2"] = (
                get_trade_cost(
                    n,
                    region,
                    ["H2 pipeline", "H2 pipeline (Kernnetz)", "H2 pipeline retrofitted"],
                )
                / 1e9
            )
            # Total Energy System Cost in billion EUR2020/yr
            tsc_sum, tsc = get_tsc(n, region)

            var["Total Energy System Cost|EU"] = (
                n.statistics.capex().sum() + n.statistics.opex().sum()
            ) / 1e9

            biomass_import_ind = tsc[tsc.index.str.contains("biomass transported")].index
            var["Total Energy System Cost|Trade|Biomass"] = tsc.loc[biomass_import_ind, "opex"].sum().sum() / 1e9
            tsc.loc[biomass_import_ind, "opex"] = 0

            gas_import_ind = tsc[tsc.carrier == "gas primary"].index
            var["Total Energy System Cost|Trade|Gas"] = tsc.loc[gas_import_ind, "opex"].sum().sum() / 1e9
            tsc.loc[gas_import_ind, "opex"] = 0

            oil_import_ind = tsc[tsc.carrier == "oil primary"].index
            var["Total Energy System Cost|Trade|Oil"] = tsc.loc[oil_import_ind, "opex"].sum().sum() / 1e9
            tsc.loc[oil_import_ind, "opex"] = 0

            tsc_all[scenario] = tsc

            # coal imports
            lignite_ind = n.links[(n.links.carrier=="lignite") & (n.links.bus1.str.contains("DE"))].index
            lignite_cost = (n.links_t.p0[lignite_ind].sum(axis=1) * n.buses_t.marginal_price["EU lignite"]).sum() / 1e9

            coal_ind = n.links[(n.links.carrier=="coal") & (n.links.bus1.str.contains("DE"))].index
            coal_cost = (n.links_t.p0[coal_ind].sum(axis=1) * n.buses_t.marginal_price["EU coal"]).sum() / 1e9

            var["Total Energy System Cost|Trade|Coal"] = lignite_cost + coal_cost

            var["Total Energy System Cost"] = (
                tsc_sum.sum().sum() / 1e9 
                + var["Total Energy System Cost|Trade|Electricity"] 
                + var["Total Energy System Cost|Trade|eFuels"]
                + var["Total Energy System Cost|Trade|H2"]
                + var["Total Energy System Cost|Trade|Coal"]
            )

            var["Total Energy System Cost|Trade"] = (
                var["Total Energy System Cost|Trade|Electricity"]
                + var["Total Energy System Cost|Trade|eFuels"]
                + var["Total Energy System Cost|Trade|H2"]
                + var["Total Energy System Cost|Trade|Biomass"]
                + var["Total Energy System Cost|Trade|Gas"]
                + var["Total Energy System Cost|Trade|Oil"]
                + var["Total Energy System Cost|Trade|Coal"]
            )

            var["Total Energy System Cost|Trade|Fuels"] = var["Total Energy System Cost|Trade|eFuels"] + var["Total Energy System Cost|Trade|Oil"]

            var["Total Energy System Cost|Non Trade"] = tsc[["capex", "opex"]].sum().sum() / 1e9

            assert abs(
                var["Total Energy System Cost"]
                - (
                    var["Total Energy System Cost|Non Trade"]
                    + var["Total Energy System Cost|Trade"]
                )
            ) < 1e-6

            # CAPEX & OPEX breakdown
            df = tsc 
            # Apply mappings
            df['bus'] = df.apply(lambda row: get_component_bus(row, n), axis=1)
            df['sector'] = df['bus'].apply(lambda x: map_bus_to_sector(x, n, bus_carrier_to_sector))

            # Sum by sector
            sector_costs = df.groupby('sector')[['capex', 'opex']].sum()

            for sector in ["Electricity", "Heat" ,"H2", "Fuels", "Gas", "Biomass"]:
                var["Total Energy System Cost|CAPEX|"+sector] = sector_costs.loc[sector, "capex"] / 1e9
                var["Total Energy System Cost|OPEX|"+sector] = sector_costs.loc[sector, "opex"] / 1e9

            other = [s for s in sector_costs.index if s not in ['Electricity', 'Heat' ,'H2', "Fuels", "Gas", "Biomass"]]
            var["Total Energy System Cost|CAPEX|Other"] = sector_costs.loc[other, "capex"].sum() / 1e9
            var["Total Energy System Cost|OPEX|Other"] = sector_costs.loc[other, "opex"].sum() / 1e9

            var["Total Energy System Cost|Trade|Other"] = (
                + var["Total Energy System Cost|Trade|Coal"]
            )
            
            var_all[scenario] = var

        df_to_png(round(var_all, 2), output_dir / f"system_cost_{year}.png")

        index_keep = [
            'Total Energy System Cost|CAPEX|Electricity',
            'Total Energy System Cost|CAPEX|Heat',
            'Total Energy System Cost|CAPEX|H2',
            'Total Energy System Cost|CAPEX|Fuels',
            'Total Energy System Cost|CAPEX|Gas',
            'Total Energy System Cost|CAPEX|Biomass',
            'Total Energy System Cost|CAPEX|Other',
            'Total Energy System Cost|OPEX|Electricity',
            'Total Energy System Cost|OPEX|Heat',
            'Total Energy System Cost|OPEX|H2',
            'Total Energy System Cost|OPEX|Fuels',
            'Total Energy System Cost|OPEX|Gas',
            'Total Energy System Cost|OPEX|Biomass',
            'Total Energy System Cost|OPEX|Other',
            'Total Energy System Cost|Trade|Electricity',
            'Total Energy System Cost|Trade|H2',
            'Total Energy System Cost|Trade|Fuels',
            'Total Energy System Cost|Trade|Gas',
            'Total Energy System Cost|Trade|Biomass',
            'Total Energy System Cost|Trade|Other',
        ]
        var_plot = pd.DataFrame(index=index_keep, columns=scenarios)

        for s in scenarios:
            var_plot[s] = var_all[s]

        assert np.isclose(var_plot.sum() , var_all.loc["Total Energy System Cost"]).all()

        df_to_png(round(var_plot, 2), output_dir / f"system_cost_plotted_{year}.png")

        fig, ax = plot_energy_system_cost_comparison(
            var_plot, 
            scenarios=scenarios,
            save_path=output_dir / f"total_energy_system_cost_by_scenario_{year}.png",
            colors=sector_colors
        )
        
        # System cost per sector and year
        for sector in ["Electricity", "Heat", "H2", "Fuels", "Gas", "Biomass", "Other"]:
            res = pd.DataFrame()
            for scenario in scenarios:

                df = tsc_all[scenario]
                res[scenario] = df[df.sector == sector].groupby(by="carrier").sum()[["capex"]]

            df = round(res/1e9,2)
            df_to_png(df[(df > 1).any(axis=1)], output_dir / f"system_cost_{sector}_{year}.png")
            df[(df > 1).any(axis=1)]

    ### TRADE ###

    plot_vars = {
        "Electricity": "Trade|Secondary Energy|Electricity|Volume",
        "Gas": "Primary Energy|Gas",
        "Oil": "Primary Energy|Oil",
        "Hydrogen": "Trade|Secondary Energy|Hydrogen|Volume", 
        "eFuels": "Trade|Secondary Energy|Efuels|Volume",
        "Biomass": "Trade|Primary Energy|Biomass|Net Imports",
    }

    bar_plot_variables(variables, 
                        scenarios = scenarios, 
                        years = planning_horizons,
                        tech_colors=dict(sector_colors, **tech_colors),
                        plot_vars=plot_vars,
                        sign_flip_vars=["Oil", "Gas", "Biomass"],
                        output_dir=output_dir / "trade_volume")
    
    plot_vars = {
        "eFuels": "Trade|Secondary Energy|Efuels|Volume",
        "Renewable Gas": "Trade|Secondary Energy|Efuels|Renewable Gas|Volume",
        "Renewable Oil": "Trade|Secondary Energy|Efuels|Renewable Oil|Volume",
        "Methanol": "Trade|Secondary Energy|Efuels|Methanol|Volume",
    }

    bar_plot_variables(variables, 
                        scenarios = scenarios, 
                        years = planning_horizons,
                        tech_colors=dict(sector_colors, **tech_colors),
                        plot_vars=plot_vars,
                        sign_flip_vars=["Oil"],
                        output_dir=output_dir / "trade_volume_efuels")
    
    plot_vars = {
        "Electricity": "Total Energy System Cost|Trade|Electricity",
        "Gas": "Total Energy System Cost|Trade|Gas",
        "Oil": "Total Energy System Cost|Trade|Oil",
        "Hydrogen": "Total Energy System Cost|Trade|Hydrogen", 
        "eFuels": "Total Energy System Cost|Trade|Efuels",
        "Biomass": "Total Energy System Cost|Trade|Biomass",
    }

    bar_plot_variables(variables, 
                        scenarios = scenarios, 
                        years = planning_horizons,
                        tech_colors=dict(sector_colors, **tech_colors),
                        plot_vars=plot_vars,
                        sign_flip_vars=[],
                        output_dir=output_dir / "trade_cost")
    

    plot_vars = {
        "eFuels": "Total Energy System Cost|Trade|Efuels",
        "Renewable Gas": "Total Energy System Cost|Trade|Efuels|Renewable Gas",
        "Renewable Oil": "Total Energy System Cost|Trade|Efuels|Renewable Oil",
        "Methanol": "Total Energy System Cost|Trade|Efuels|Methanol",
    }

    bar_plot_variables(variables, 
                        scenarios = scenarios, 
                        years = planning_horizons,
                        tech_colors=dict(sector_colors, **tech_colors),
                        plot_vars=plot_vars,
                        sign_flip_vars=[],
                        output_dir=output_dir / "trade_cost_efuels")

    ### CURTAILMENT ###

    for year in planning_horizons:
        plot_curtailment(networks, 
                         scenarios,
                         year,
                         tech_colors,
                         output_dir)
