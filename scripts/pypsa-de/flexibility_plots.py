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
from flexibility_utils import tech_colors, tech_groups, year_colors_gradient, find_project_root

logger = logging.getLogger(__name__)


def plot_flex_needs(
    flex_needs_t,
    year_colors_gradient,
    output_file,
    bar_width=0.13,
    inner_spacing=0.02,
    group_spacing=0.2,
    figsize=(12, 6),
):
    """
    Plot flexibility needs by temporal granularity and year.

    Parameters
    ----------
    flex_needs_t : pandas.DataFrame
        DataFrame with years as index and granularities as columns.
    year_colors_gradient : dict
        Mapping {year: color} for bar colors.
    output_file : str
        Path to save the plot (e.g. PNG, PDF).
    bar_width : float
        Width of each bar.
    inner_spacing : float
        Spacing between bars within one group.
    group_spacing : float
        Extra spacing between groups.
    figsize : tuple
        Size of the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # X positions for granularities
    x = range(len(flex_needs_t.columns))

    # Create bars for each year
    for i, year in enumerate(flex_needs_t.index):
        x_pos = [p + i * (bar_width + inner_spacing) + p * group_spacing for p in x]
        bars = ax.bar(
            x_pos,
            flex_needs_t.loc[year],
            width=bar_width,
            label=year,
            color=year_colors_gradient[int(year)],
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Configure plot
    ax.set_xticks(
        [
            p
            + (len(flex_needs_t.index) - 1) * (bar_width + inner_spacing) / 2
            + p * group_spacing
            for p in x
        ]
    )
    ax.set_xticklabels(flex_needs_t.columns, rotation=0)
    ax.set_ylabel("TWh/a", fontsize=12)
    ax.set_title(
        "Flexibility Needs by Granularity and Year", fontsize=14, fontweight="bold"
    )
    ax.legend(title="Year", loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Save and close
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_flexibility_causes_multiyear(
    multiyear_causes_dict, tech_colors, figsize=(16, 8), save_path=None
):
    """
    Plot flexibility causes for multiple years as stacked bar charts.
    Shows 4 granularities with years as separate bars, stacked by technology.

    Parameters
    ----------
    multiyear_causes_dict : dict
        Dictionary with year as key and inflexible_df as value
        e.g. {2020: inflexible_df_2020, 2025: inflexible_df_2025, ...}
    tech_colors : dict
        Dictionary mapping technology names to colors
    figsize : tuple, default (16, 8)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """

    # Combine all years into one DataFrame
    all_results = []
    for year, inflexible_df in multiyear_causes_dict.items():
        df_year = inflexible_df.reset_index()
        df_year["Year"] = year
        all_results.append(df_year)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Extract carrier names (remove Supply_/Demand_ prefix)
    def extract_carrier_name(tech_name):
        if tech_name.startswith("Supply_"):
            return tech_name[7:]
        elif tech_name.startswith("Demand_"):
            return tech_name[7:]
        else:
            return tech_name

    combined_df["Carrier"] = combined_df["Technology"].apply(extract_carrier_name)

    # Group by Year, Granularity, and Carrier (sum supply and demand)
    df_grouped = (
        combined_df.groupby(["Year", "Granularity", "Carrier"])[
            "Contribution (TWh/year)"
        ]
        .sum()
        .reset_index()
    )

    # Flip the signs: causes should be positive, solutions should be negative
    df_grouped["Contribution (TWh/year)"] = -df_grouped["Contribution (TWh/year)"]

    # Pivot to get structure: MultiIndex(Year, Granularity) as rows, Carriers as columns
    df_pivot = df_grouped.pivot_table(
        index=["Year", "Granularity"],
        columns="Carrier",
        values="Contribution (TWh/year)",
        fill_value=0,
    )

    # Filter out very small contributions (< 1% of max absolute value)
    max_val = df_pivot.abs().max().max()
    threshold = max_val * 0.01
    significant_carriers = df_pivot.columns[(df_pivot.abs() > threshold).any()]
    df_filtered = df_pivot[significant_carriers]

    # Select granularities to plot
    granularities = ["daily", "weekly", "monthly", "annual"]
    available_grans = [
        g
        for g in granularities
        if g in df_filtered.index.get_level_values("Granularity")
    ]

    if not available_grans:
        raise ValueError("No valid granularities found in data")

    # Set up subplots - one for each granularity
    fig, axes = plt.subplots(1, len(available_grans), figsize=figsize, sharey=True)
    if len(available_grans) == 1:
        axes = [axes]

    # Title mapping
    title_map = {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "annual": "Annual",
    }

    # Get colors for carriers
    colors = [tech_colors.get(carrier, "gray") for carrier in df_filtered.columns]

    # Plot each granularity
    for i, gran in enumerate(available_grans):
        # Get data for this granularity
        gran_data = df_filtered.xs(gran, level="Granularity")

        if gran_data.empty:
            continue

        # Separate positive and negative contributions
        positive_data = gran_data.clip(lower=0)  # Causes (now positive)
        negative_data = gran_data.clip(upper=0)  # Solutions (now negative)

        # Plot positive contributions (causes - above x-axis)
        if not positive_data.empty and positive_data.sum().sum() > 0:
            positive_data.plot(
                kind="bar",
                stacked=True,
                color=colors,
                ax=axes[i],
                legend=False,
                width=0.7,
                alpha=0.8,
            )

        # Plot negative contributions (solutions - below x-axis)
        if not negative_data.empty and negative_data.sum().sum() < 0:
            negative_data.plot(
                kind="bar",
                stacked=True,
                color=colors,
                ax=axes[i],
                legend=False,
                width=0.7,
                alpha=0.8,
            )

        # Calculate total flexibility needs (sum of absolute values)
        total_flex_needs = gran_data.sum(axis=1)

        # Add single horizontal dashed line for total flexibility needs (positive only)
        for j, year in enumerate(gran_data.index):
            total = total_flex_needs.loc[year]
            # Add horizontal dashed line only at positive value
            axes[i].hlines(
                y=total,
                xmin=j - 0.35,
                xmax=j + 0.35,
                colors="black",
                linestyles="--",
                linewidth=2,
                alpha=0.8,
            )

        # Formatting
        axes[i].set_title(f"{title_map[gran]} Flexibility Needs", fontsize=14)
        axes[i].set_xlabel("Year", fontsize=12)
        axes[i].tick_params(axis="x", rotation=45, labelsize=10)
        axes[i].tick_params(axis="y", labelsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        if i == 0:
            axes[i].set_ylabel("Flexibility Contribution (TWh/a)", fontsize=12)

    # Create unified legend with technologies and total line
    handles, labels = [], []

    # Add technology legend entries
    for carrier in df_filtered.columns:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=tech_colors.get(carrier, "gray"),
                markersize=10,
            )
        )
        labels.append(carrier)

    # Add total flexibility needs line entry
    handles.append(
        plt.Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=0.8)
    )
    labels.append("Total Flexibility Needs")

    # Place legend below plots
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(8, len(labels)),
        title="Technologies",
        fontsize=10,
    )

    plt.suptitle(
        "Flexibility Needs Causes by Granularity and Year", fontsize=16, y=0.98
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()
    return fig, axes


def plot_flexibility_provision_multiyear(
    flexibility_df, tech_colors, figsize=(16, 8), save_path=None
):
    """
    Plot flexibility provision for multiple years as stacked bar charts.

    Parameters
    ----------
    flexibility_df : pd.DataFrame
        DataFrame with MultiIndex (Granularity, Year) and technology columns
    tech_colors : dict
        Dictionary mapping technology names to colors
    figsize : tuple, default (16, 8)
        Figure size
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """

    # Select granularities to plot
    granularities = ["daily", "weekly", "monthly", "annual"]
    available_grans = [
        g
        for g in granularities
        if g in flexibility_df.index.get_level_values("Granularity")
    ]

    if not available_grans:
        raise ValueError("No valid granularities found in data")

    # Set up subplots - one for each granularity
    fig, axes = plt.subplots(1, len(available_grans), figsize=figsize, sharey=True)
    if len(available_grans) == 1:
        axes = [axes]

    # Title mapping
    title_map = {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "annual": "Annual",
    }

    # Get colors for technologies
    colors = [tech_colors.get(tech, "gray") for tech in flexibility_df.columns]

    # Plot each granularity
    for i, gran in enumerate(available_grans):
        # Get data for this granularity
        gran_data = flexibility_df.xs(gran, level="Granularity")

        if gran_data.empty:
            continue

        # Separate positive and negative contributions
        positive_data = gran_data.clip(lower=0)  # Solutions (positive)
        negative_data = gran_data.clip(upper=0)  # Any negative contributions

        # Plot positive contributions (solutions - above x-axis)
        if not positive_data.empty and positive_data.sum().sum() > 0:
            positive_data.plot(
                kind="bar",
                stacked=True,
                color=colors,
                ax=axes[i],
                legend=False,
                width=0.7,
                alpha=0.8,
            )

        # Plot negative contributions (below x-axis)
        if not negative_data.empty and negative_data.sum().sum() < 0:
            negative_data.plot(
                kind="bar",
                stacked=True,
                color=colors,
                ax=axes[i],
                legend=False,
                width=0.7,
                alpha=0.8,
            )

        # Calculate total flexibility provision (sum of absolute values)
        total_flex_provision = gran_data.sum(axis=1)

        # Add single horizontal dashed line for total flexibility provision
        for j, year in enumerate(gran_data.index):
            total = total_flex_provision.loc[year]
            # Add horizontal dashed line only at positive value
            axes[i].hlines(
                y=total,
                xmin=j - 0.35,
                xmax=j + 0.35,
                colors="black",
                linestyles="--",
                linewidth=2,
                alpha=0.8,
            )

        # Formatting
        axes[i].set_title(f"{title_map[gran]} Flexibility Provision", fontsize=14)
        axes[i].set_xlabel("Year", fontsize=12)
        axes[i].tick_params(axis="x", rotation=45, labelsize=10)
        axes[i].tick_params(axis="y", labelsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        if i == 0:
            axes[i].set_ylabel("Flexibility Contribution (TWh/a)", fontsize=12)

    # Create unified legend with technologies and total line
    handles, labels = [], []

    # Add technology legend entries
    for tech in flexibility_df.columns:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=tech_colors.get(tech, "gray"),
                markersize=10,
            )
        )
        labels.append(tech)

    # Add total flexibility provision line entry
    handles.append(
        plt.Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=0.8)
    )
    labels.append("Total Flexibility Provision")

    # Place legend below plots
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(8, len(labels)),
        title="Technologies",
        fontsize=10,
    )

    plt.suptitle("Flexibility Provision by Granularity and Year", fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()

    return fig, axes


def plot_flexibility_needs_map(
    flex_needs_per_node_year,
    load_buses,
    onshore_regions,
    year,
    output_path,
    region="DE",
    extent=None,
    figsize=(16, 12),
    dpi=300,
    cmap="viridis_r",
):
    """
    Plot flexibility needs maps for different time periods.

    Parameters
    ----------
    flex_needs_per_node_year : pd.DataFrame
        Flexibility needs data with periods as index
    load_buses : pd.Series or pd.DataFrame
        Load data for normalization
    onshore_regions : gpd.GeoDataFrame
        Geographic regions to plot
    year : int or str
        Year for the plot title
    output_path : str
        Path to save the figure
    region : str, optional
        Region code to filter (default: "DE")
    extent : list, optional
        Map extent [lon_min, lon_max, lat_min, lat_max] (default: Germany extent)
    figsize : tuple, optional
        Figure size (default: (16, 12))
    dpi : int, optional
        Resolution for saved figure (default: 300)
    cmap : str, optional
        Colormap name (default: "viridis_r")
    """
    if extent is None:
        extent = [5.5, 15.5, 47, 56]  # Default Germany extent

    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    display_projection = ccrs.EqualEarth()

    # Prepare data
    df = onshore_regions.copy()
    for period in ["daily", "weekly", "monthly", "annual"]:
        df[f"flex_{period}"] = pd.to_numeric(
            (flex_needs_per_node_year / load_buses.values.T).loc[period],
            errors="coerce",
        )

    df_region = df[df.index.str.contains(region)].copy()

    # Create subplots
    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": display_projection}, figsize=figsize
    )
    axes = axes.flatten()

    periods = ["daily", "weekly", "monthly", "annual"]
    titles = [
        "Daily Flexibility Needs",
        "Weekly Flexibility Needs",
        "Monthly Flexibility Needs",
        "Annual Flexibility Needs",
    ]

    for i, (period, title) in enumerate(zip(periods, titles)):
        ax = axes[i]

        # Calculate vmin/vmax for this period
        vmin, vmax = (
            df_region[f"flex_{period}"].min(),
            df_region[f"flex_{period}"].max(),
        )

        # Add map features
        ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.coastlines(edgecolor="black", linewidth=0.5)
        ax.set_facecolor("white")
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.set_title(title, pad=15)

        # Plot data
        df_region.to_crs(display_projection.proj4_init).plot(
            column=f"flex_{period}",
            ax=ax,
            linewidth=0.05,
            edgecolor="grey",
            legend=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        # Set extent and aspect
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.set_aspect(aspect_ratio)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f"{title} (normalised by load)", rotation=270, labelpad=15)

    # Add overall title with year
    fig.suptitle(f"Flexibility Needs - {year}", fontsize=16, y=0.98)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return fig, axes


def plot_flexibility_causes_map(
    inflex_cause_bus,
    buses_de,
    bus_coords,
    onshore_regions,
    tech_colors,
    year,
    output_path,
    scale_factor=1e4,
    extent=None,
    figsize=(16, 14),
    dpi=300,
    size_legend_values=None,
):
    """
    Plot flexibility causes maps for different time periods with pie charts at each bus.

    Parameters
    ----------
    inflex_cause_bus : dict
        Dictionary with bus names as keys and inflexibility cause data as values
    buses_de : list or Index
        List of bus names to plot
    bus_coords : pd.DataFrame
        DataFrame with bus coordinates (columns: 'x', 'y')
    onshore_regions : gpd.GeoDataFrame
        Geographic regions to plot as background
    tech_colors : dict
        Dictionary mapping technology names to colors
    year : int or str
        Year for the plot title
    output_path : str or Path
        Path to save the figure
    scale_factor : float, optional
        Scale factor for circle sizes (default: 1e4)
    extent : list, optional
        Map extent [lon_min, lon_max, lat_min, lat_max] (default: Germany extent)
    figsize : tuple, optional
        Figure size (default: (16, 14))
    dpi : int, optional
        Resolution for saved figure (default: 300)
    size_legend_values : list, optional
        Values (in TWh) for size legend circles (default: [5, 25])
    """
    if extent is None:
        extent = [5.5, 15.5, 47, 55.5]  # Default Germany extent

    if size_legend_values is None:
        size_legend_values = [5, 25]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create 4 subplots for different time scales
    display_projection = ccrs.EqualEarth()
    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": display_projection}, figsize=figsize
    )
    axes = axes.flatten()

    time_scales = ["daily", "weekly", "monthly", "annual"]
    titles = [
        "Daily Flexibility Causes",
        "Weekly Flexibility Causes",
        "Monthly Flexibility Causes",
        "Annual Flexibility Causes",
    ]

    # Collect all used technologies across all subplots for unified legend
    all_used_techs_with_colors = {}

    for idx, (time_scale, title) in enumerate(zip(time_scales, titles)):
        ax = axes[idx]

        # Set up map
        ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.coastlines(edgecolor="black", linewidth=0.5)
        ax.set_facecolor("white")
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.set_title(title, pad=15)

        # Add region boundaries
        onshore_regions.to_crs(display_projection.proj4_init).plot(
            ax=ax, facecolor="none", edgecolor="grey", linewidth=0.3, alpha=0.7
        )

        # Plot circles for each bus
        for bus_name in buses_de:
            if bus_name in inflex_cause_bus:
                # Get inflexibility data from dictionary for this time scale
                bus_data = inflex_cause_bus[bus_name].loc[time_scale]

                # Clean tech names and get absolute values
                cleaned_data = {}
                for tech in bus_data.index:
                    clean_name = tech.replace("Supply_", "").replace("Demand_", "")
                    value = bus_data.loc[tech, "Contribution (TWh/year)"]
                    # Only use negative values (actual inflexibility causes)
                    if float(value) < -1e-6:  # Only significant negative values
                        cleaned_data[clean_name] = abs(float(value))

                if not cleaned_data:
                    continue

                # Calculate circle size and position
                total_inflex = sum(cleaned_data.values())
                radius = np.sqrt(total_inflex) * scale_factor

                lon, lat = bus_coords.loc[bus_name, ["x", "y"]]
                x, y = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())

                # Create pie chart
                techs = list(cleaned_data.keys())
                values = list(cleaned_data.values())
                colors = [tech_colors.get(tech, "gray") for tech in techs]

                # Track used technologies and their colors
                for tech, color in zip(techs, colors):
                    if color and color != "":
                        all_used_techs_with_colors[tech] = color

                # Draw pie wedges
                angles = np.array(values) / sum(values) * 2 * np.pi
                start_angle = 0

                # If only one technology, draw a full circle
                if len(values) == 1:
                    circle = plt.Circle(
                        (x, y),
                        radius,
                        facecolor=colors[0],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.add_patch(circle)
                else:
                    # Draw pie wedges for multiple technologies
                    for angle, color in zip(angles, colors):
                        theta1 = np.degrees(start_angle)
                        theta2 = np.degrees(start_angle + angle)

                        wedge = plt.matplotlib.patches.Wedge(
                            (x, y),
                            radius,
                            theta1,
                            theta2,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax.add_patch(wedge)
                        start_angle += angle

        # Set map extent
        ax.set_extent(extent, ccrs.PlateCarree())

        # Add size legend to each subplot
        size_legend_labels = [f"{val} TWh" for val in size_legend_values]
        size_legend_radii = [np.sqrt(val) * scale_factor for val in size_legend_values]

        # Position for size legend (bottom left of each subplot)
        legend_x = 0.03
        legend_y = 0.18
        legend_spacing = 0.1

        for i, (radius, label) in enumerate(zip(size_legend_radii, size_legend_labels)):
            y_pos = legend_y - i * legend_spacing
            # Add circle to show size
            circle = plt.Circle(
                (legend_x + 0.03, y_pos),
                radius / 1e6,
                facecolor="lightgrey",
                edgecolor="black",
                linewidth=0.5,
                transform=ax.transAxes,
            )
            ax.add_patch(circle)
            # Add text label
            ax.text(
                legend_x + 0.12,
                y_pos,
                label,
                transform=ax.transAxes,
                verticalalignment="center",
                fontsize=7,
            )

        # Add title for size legend
        ax.text(
            legend_x,
            legend_y + 0.03,
            "Circle Size\n(Flexibility needs)",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8,
        )

    # Add unified technology legend below all subplots
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor=color, label=tech)
        for tech, color in all_used_techs_with_colors.items()
    ]

    # Create legend below the subplots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(legend_elements), 8),  # Max 8 columns
        title="Technologies",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    # Add overall title with year
    fig.suptitle(f"Flexibility Causes - {year}", fontsize=16, y=0.98)

    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.12)
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return fig, axes


def plot_flexibility_provision_map(
    flex_contribution_bus,
    buses_de,
    bus_coords,
    onshore_regions,
    tech_colors,
    tech_groups,
    year,
    output_path,
    scale_factor=1e4,
    extent=None,
    figsize=(16, 14),
    dpi=300,
    size_legend_values=None,
    small_tech_threshold=0.1,
    other_threshold=0.01,
):
    """
    Plot flexibility provision maps for different time periods with pie charts at each bus.

    Parameters
    ----------
    flex_contribution_bus : dict
        Dictionary with bus names as keys and flexibility contribution data as values
    buses_de : list or Index
        List of bus names to plot
    bus_coords : pd.DataFrame
        DataFrame with bus coordinates (columns: 'x', 'y')
    onshore_regions : gpd.GeoDataFrame
        Geographic regions to plot as background
    tech_colors : dict
        Dictionary mapping technology names to colors
    tech_groups : dict
        Dictionary for grouping technologies by keywords
    year : int or str
        Year for the plot title
    output_path : str or Path
        Path to save the figure
    scale_factor : float, optional
        Scale factor for circle sizes (default: 1e4)
    extent : list, optional
        Map extent [lon_min, lon_max, lat_min, lat_max] (default: Germany extent)
    figsize : tuple, optional
        Figure size (default: (16, 14))
    dpi : int, optional
        Resolution for saved figure (default: 300)
    size_legend_values : list, optional
        Values (in TWh) for size legend circles (default: [5, 25])
    small_tech_threshold : float, optional
        Threshold (TWh) below which technologies are grouped into "Other" (default: 0.1)
    other_threshold : float, optional
        Minimum total value for "Other" category to be included (default: 0.01)
    """
    if extent is None:
        extent = [5.5, 15.5, 47, 55.5]  # Default Germany extent

    if size_legend_values is None:
        size_legend_values = [5, 25]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create 4 subplots for different time scales
    display_projection = ccrs.EqualEarth()
    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": display_projection}, figsize=figsize
    )
    axes = axes.flatten()

    time_scales = ["daily", "weekly", "monthly", "annual"]
    titles = [
        "Daily Flexibility Provision",
        "Weekly Flexibility Provision",
        "Monthly Flexibility Provision",
        "Annual Flexibility Provision",
    ]

    # Collect all used technologies across all subplots for unified legend
    all_used_techs_with_colors = {}

    for idx, (time_scale, title) in enumerate(zip(time_scales, titles)):
        ax = axes[idx]

        # Set up map
        ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.coastlines(edgecolor="black", linewidth=0.5)
        ax.set_facecolor("white")
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.set_title(title, pad=15)

        # Add region boundaries
        onshore_regions.to_crs(display_projection.proj4_init).plot(
            ax=ax, facecolor="none", edgecolor="grey", linewidth=0.3, alpha=0.7
        )

        # Plot circles for each bus
        for bus_name in buses_de:
            if bus_name in flex_contribution_bus:
                # Get flexibility contribution data for this time scale
                bus_data = flex_contribution_bus[bus_name].loc[time_scale]

                # Create a DataFrame for aggregation (technologies as index)
                tech_data = pd.DataFrame({"value": bus_data["Contribution (TWh/year)"]})
                tech_data.index = [
                    tech.replace("Supply_", "").replace("Demand_", "")
                    for tech in bus_data.index
                ]

                # Apply keyword-based aggregation
                aggregated_data = aggregate_by_keywords(tech_data, tech_groups)

                # Filter for positive values (actual flexibility provision) and significant values
                cleaned_data = {}
                for tech, value in aggregated_data["value"].items():
                    if float(value) > 1e-6:  # Only significant positive values
                        cleaned_data[tech] = float(value)

                # Group small contributions into "Other"
                if cleaned_data:
                    small_techs = {
                        k: v
                        for k, v in cleaned_data.items()
                        if v < small_tech_threshold
                    }
                    large_techs = {
                        k: v
                        for k, v in cleaned_data.items()
                        if v >= small_tech_threshold
                    }

                    if (
                        small_techs and len(large_techs) > 0
                    ):  # Only create "Other" if there are also large techs
                        other_sum = sum(small_techs.values())
                        if (
                            other_sum > other_threshold
                        ):  # Only add "Other" if non-negligible
                            large_techs["Other"] = other_sum
                        cleaned_data = large_techs

                if not cleaned_data:
                    continue

                # Calculate circle size and position
                total_flex = sum(cleaned_data.values())
                radius = np.sqrt(total_flex) * scale_factor

                lon, lat = bus_coords.loc[bus_name, ["x", "y"]]
                x, y = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())

                # Create pie chart
                techs = list(cleaned_data.keys())
                values = list(cleaned_data.values())
                colors = [tech_colors.get(tech, "gray") for tech in techs]

                # Track used technologies and their colors
                for tech, color in zip(techs, colors):
                    if color and color != "":
                        all_used_techs_with_colors[tech] = color

                # Draw pie wedges
                angles = np.array(values) / sum(values) * 2 * np.pi
                start_angle = 0

                # If only one technology, draw a full circle
                if len(values) == 1:
                    circle = plt.Circle(
                        (x, y),
                        radius,
                        facecolor=colors[0],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.add_patch(circle)
                else:
                    # Draw pie wedges for multiple technologies
                    for angle, color in zip(angles, colors):
                        theta1 = np.degrees(start_angle)
                        theta2 = np.degrees(start_angle + angle)

                        wedge = plt.matplotlib.patches.Wedge(
                            (x, y),
                            radius,
                            theta1,
                            theta2,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax.add_patch(wedge)
                        start_angle += angle

        # Set map extent
        ax.set_extent(extent, ccrs.PlateCarree())

        # Add size legend to each subplot
        size_legend_labels = [f"{val} TWh" for val in size_legend_values]
        size_legend_radii = [np.sqrt(val) * scale_factor for val in size_legend_values]

        # Position for size legend (bottom left of each subplot)
        legend_x = 0.03
        legend_y = 0.18
        legend_spacing = 0.1

        for i, (radius, label) in enumerate(zip(size_legend_radii, size_legend_labels)):
            y_pos = legend_y - i * legend_spacing
            # Add circle to show size
            circle = plt.Circle(
                (legend_x + 0.03, y_pos),
                radius / 1e6,
                facecolor="lightgrey",
                edgecolor="black",
                linewidth=0.5,
                transform=ax.transAxes,
            )
            ax.add_patch(circle)
            # Add text label
            ax.text(
                legend_x + 0.12,
                y_pos,
                label,
                transform=ax.transAxes,
                verticalalignment="center",
                fontsize=7,
            )

        # Add title for size legend
        ax.text(
            legend_x,
            legend_y + 0.03,
            "Circle Size\n(Flexibility)",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8,
        )

    # Add unified technology legend below all subplots
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor=color, label=tech)
        for tech, color in all_used_techs_with_colors.items()
    ]

    # Create legend below the subplots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(legend_elements), 8),  # Max 8 columns
        title="Technologies",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    # Add overall title with year
    fig.suptitle(f"Flexibility Provision - {year}", fontsize=16, y=0.98)

    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.12)
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return fig, axes


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        project_root = find_project_root()
        os.chdir(project_root)

        snakemake = mock_snakemake(
            "flexibility_plots",
            simpl="",
            clusters=27,
            opts="",
            sector_opts="None",
            run="HighFlex",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons

    # Make sure output directory exists
    os.makedirs(snakemake.params.output_dir, exist_ok=True)

    # Load networks
    networks = {int(fn[-7:-3]): pypsa.Network(fn) for fn in snakemake.input.networks}

    # Load results
    flex_needs = pd.read_csv(snakemake.input.flex_needs, index_col=0)
    with open(snakemake.input.flex_causes_raw, "rb") as f:
        flex_causes_raw = pickle.load(f)
    flex_contributions_clean = pd.read_csv(
        snakemake.input.flex_contributions_clean, index_col=[0, 1]
    )
    flex_needs_per_node = pd.read_pickle(snakemake.input.flex_needs_per_node)
    flex_causes_per_node = pd.read_pickle(snakemake.input.flex_causes_per_node)
    flex_contributions_per_node = pd.read_pickle(
        snakemake.input.flex_contributions_per_node
    )
    logger.info("Loaded results.")

    ####### Plotting

    # Plot flexibility needs
    flex_needs_t = flex_needs.transpose()
    logger.info("Plotting flexibility needs...")

    plot_flex_needs(
        flex_needs_t,
        year_colors_gradient,
        snakemake.output.flex_needs_plot,
    )

    # Plot flexibility causes
    logger.info("Plotting flexibility causes...")

    plot_flexibility_causes_multiyear(
        flex_causes_raw,
        tech_colors,
        figsize=(20, 10),
        save_path=snakemake.output.flex_causes_plot,
    )

    # Plotting flexibility contributions
    logger.info("Plotting flexibility contributions...")

    plot_flexibility_provision_multiyear(
        flex_contributions_clean,
        tech_colors,
        figsize=(20, 10),
        save_path=snakemake.output.flex_contributions_plot,
    )

    # Plotting flexibility needs per node

    # load regions
    onshore_regions = gpd.read_file(
        snakemake.input.regions_onshore_clustered
    ).set_index("name")

    region = "DE"
    bus_carrier = ["AC"]
    n = networks[planning_horizons[0]]
    buses_de = n.buses[
        (n.buses.index.str[:2] == region) & (n.buses.carrier.isin(bus_carrier))
    ].index
    load_buses = pd.DataFrame(index=buses_de, columns=["Load (TWh/year)"])

    for year in planning_horizons:
        n = networks[year]
        flex_needs_per_node_year = flex_needs_per_node[year]

        for bus in buses_de:
            load_buses.loc[bus, "Load (TWh/year)"] = (
                n.loads_t.p[
                    n.loads[n.loads.bus == bus + " low voltage"].index
                ].values.sum()
                / 1e6
            )

        logger.info(f"Plotting flexibility needs map for year {year}...")

        plot_flexibility_needs_map(
            flex_needs_per_node_year=flex_needs_per_node[year],
            load_buses=load_buses,
            onshore_regions=onshore_regions,
            year=year,
            output_path=f"{snakemake.params.output_dir}/flex_needs_map_{year}.png",
        )

    for year in planning_horizons:
        logger.info(f"Plotting flexibility causes map for year {year}...")

        plot_flexibility_causes_map(
            inflex_cause_bus=flex_causes_per_node[year],
            buses_de=buses_de,
            bus_coords=n.buses.loc[buses_de, ["x", "y"]],
            onshore_regions=onshore_regions,
            tech_colors=tech_colors,
            year=year,
            output_path=f"{snakemake.params.output_dir}/flex_causes_maps_{year}.png",
        )

    for year in planning_horizons:
        logger.info(f"Plotting flexibility contributions map for year {year}...")

        plot_flexibility_provision_map(
            flex_contribution_bus=flex_contributions_per_node[year],
            buses_de=buses_de,
            bus_coords=n.buses.loc[buses_de, ["x", "y"]],
            onshore_regions=onshore_regions,
            tech_colors=tech_colors,
            tech_groups=tech_groups,
            year=year,
            output_path=f"{snakemake.params.output_dir}/flex_contributions_maps_{year}.png",
        )
