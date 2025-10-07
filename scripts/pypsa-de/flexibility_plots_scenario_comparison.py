import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pypsa
import pickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from flexibility_utils import tech_colors, find_project_root
from flexibility_analysis import aggregate_by_keywords
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def plot_flex_needs_comparison(
    flex_needs_dict,
    output_file,
    granularities=None,
    bar_width=0.2,
    inner_spacing=0.03,
    figsize=(20, 6),
    show_values=True,
    colormaps=None,
):
    """
    Plot flexibility needs comparison across multiple scenarios.
    Groups by granularity (Daily, Weekly, Monthly, Yearly) with years on x-axis.
    
    Parameters
    ----------
    flex_needs_dict : dict
        Dictionary with {scenario: DataFrame} where DataFrame has granularities as index
        and years as columns.
    output_file : str
        Path to save the plot (e.g. PNG, PDF).
    granularities : list, optional
        List of granularities to plot. If None, uses all rows.
    bar_width : float
        Width of each bar.
    inner_spacing : float
        Spacing between bars within one group.
    figsize : tuple
        Size of the figure.
    show_values : bool
        Whether to show value labels on bars.
    colormaps : list, optional
        List of matplotlib colormap names (e.g., ['Blues', 'Oranges', 'Greens']).
        If None, uses default colormaps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Default colormaps for scenarios (distinct but gradient within each)
    if colormaps is None:
        colormaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 
                     'YlOrBr', 'PuRd', 'GnBu', 'OrRd', 'BuPu']
    
    # Get first scenario to determine structure
    first_scenario = list(flex_needs_dict.keys())[0]
    first_df = flex_needs_dict[first_scenario]
    
    # Determine granularities to plot
    if granularities is None:
        granularities = first_df.index.tolist()
    
    # Get all years (columns)
    years = first_df.columns.tolist()
    n_scenarios = len(flex_needs_dict)
    n_years = len(years)
    n_granularities = len(granularities)
    
    # Find global min and max for shared y-axis
    all_values = []
    for scenario, df in flex_needs_dict.items():
        for granularity in granularities:
            all_values.extend(df.loc[granularity].values)
    y_max = max(all_values) * 1.15  # Add 15% headroom for labels
    
    # Create subplots - one for each granularity
    fig, axes = plt.subplots(1, n_granularities, figsize=figsize, sharey=True)
    
    # Handle case of single granularity
    if n_granularities == 1:
        axes = [axes]
    
    # Plot each granularity
    for gran_idx, granularity in enumerate(granularities):
        ax = axes[gran_idx]
        
        # X positions for years
        x = np.arange(n_years)
        
        # Create bars for each scenario
        for scen_idx, (scenario, df) in enumerate(flex_needs_dict.items()):
            # Get colormap for this scenario
            cmap = plt.colormaps.get_cmap(colormaps[scen_idx % len(colormaps)])
            
            # Get values for this granularity across all years
            values = df.loc[granularity].values
            
            # Create bars with gradient colors across years
            for year_idx, (year, value) in enumerate(zip(years, values)):
                # Color gradient from light (0.3) to dark (0.9)
                color_intensity = 0.3 + 0.6 * (year_idx / max(n_years - 1, 1))
                bar_color = cmap(color_intensity)
                
                x_pos = x[year_idx] + scen_idx * (bar_width + inner_spacing)
                
                bar = ax.bar(
                    x_pos,
                    value,
                    width=bar_width,
                    label=scenario if year_idx == 0 else "",  # Only label first bar
                    color=bar_color,
                    alpha=0.9,
                    edgecolor='white',
                    linewidth=0.5,
                )
                
                # Add value labels on top of bars
                if show_values and value > 0:
                    fontsize = 7 if n_years * n_scenarios > 20 else 8
                    ax.text(
                        x_pos + bar_width / 2.0,
                        value,
                        f"{value:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=fontsize,
                        rotation=0,
                    )
        
        # Configure subplot
        ax.set_xticks(x + (n_scenarios - 1) * (bar_width + inner_spacing) / 2)
        ax.set_xticklabels(years, rotation=0, fontsize=10)
        ax.set_xlabel("Scenario Year", fontsize=11, fontweight='bold')
        ax.set_title(f"{granularity} flexibility", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set shared y-axis limits
        ax.set_ylim(0, y_max)
        
        # Only add y-label to first subplot
        if gran_idx == 0:
            ax.set_ylabel("TWh/a", fontsize=12, fontweight='bold')
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Weather Year",
        title_fontsize=11,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)  # Make room for legend
    
    # Save and close
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved flexibility needs comparison plot to {output_file}")



def plot_flexibility_causes_scenario_comparison(
    flex_causes_dict,
    tech_colors,
    output_file,
    granularities=None,
    figsize=(18, 8),
    colormaps=None,
):
    """
    Plot flexibility causes comparison across multiple scenarios.
    Shows stacked bar charts with scenario-specific patterns.
    
    Parameters
    ----------
    flex_causes_dict : dict
        Dictionary with {scenario: multiyear_causes_dict} where multiyear_causes_dict
        is {year: inflexible_df}
    tech_colors : dict
        Dictionary mapping technology names to colors
    output_file : str
        Path to save the plot
    granularities : list, optional
        List of granularities to plot
    figsize : tuple
        Figure size
    colormaps : list, optional
        List of matplotlib colormap names for scenario hatching/patterns
    """
    
    # Default hatching patterns for scenarios
    patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo']
    
    # Combine all scenarios and years
    all_results = []
    for scenario, multiyear_dict in flex_causes_dict.items():
        for year, inflexible_df in multiyear_dict.items():
            df_temp = inflexible_df.reset_index()
            df_temp["Year"] = year
            df_temp["Scenario"] = scenario
            all_results.append(df_temp)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Extract carrier names
    def extract_carrier_name(tech_name):
        if tech_name.startswith("Supply_"):
            return tech_name[7:]
        elif tech_name.startswith("Demand_"):
            return tech_name[7:]
        return tech_name
    
    combined_df["Carrier"] = combined_df["Technology"].apply(extract_carrier_name)
    
    # Group by Scenario, Year, Granularity, and Carrier
    df_grouped = (
        combined_df.groupby(["Scenario", "Year", "Granularity", "Carrier"])
        ["Contribution (TWh/year)"]
        .sum()
        .reset_index()
    )
    
    # Flip signs: causes positive, solutions negative
    df_grouped["Contribution (TWh/year)"] = -df_grouped["Contribution (TWh/year)"]
    
    # Pivot
    df_pivot = df_grouped.pivot_table(
        index=["Scenario", "Year", "Granularity"],
        columns="Carrier",
        values="Contribution (TWh/year)",
        fill_value=0,
    )
    
    # Filter small contributions
    max_val = df_pivot.abs().max().max()
    threshold = max_val * 0.01
    significant_carriers = df_pivot.columns[(df_pivot.abs() > threshold).any()]
    df_filtered = df_pivot[significant_carriers]
    
    # Select granularities
    if granularities is None:
        granularities = ["daily", "weekly", "monthly", "annual"]
    available_grans = [
        g for g in granularities
        if g in df_filtered.index.get_level_values("Granularity")
    ]
    
    # Setup subplots
    fig, axes = plt.subplots(1, len(available_grans), figsize=figsize, sharey=True)
    if len(available_grans) == 1:
        axes = [axes]
    
    title_map = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "annual": "Annual"}
    
    scenarios = list(flex_causes_dict.keys())
    n_scenarios = len(scenarios)
    
    # Find global y limits for all subplots
    all_y_values = []
    
    # Plot each granularity
    for i, gran in enumerate(available_grans):
        gran_data = df_filtered.xs(gran, level="Granularity")
        
        if gran_data.empty:
            continue
        
        # Get unique years
        years = sorted(gran_data.index.get_level_values("Year").unique())
        n_years = len(years)
        
        # Bar positions
        bar_width = 0.8 / n_scenarios
        x = np.arange(n_years)
        
        # Plot each scenario
        for scen_idx, scenario in enumerate(scenarios):
            if scenario not in gran_data.index.get_level_values("Scenario"):
                continue
            
            scenario_data = gran_data.xs(scenario, level="Scenario")
            
            # Separate positive and negative
            positive_data = scenario_data.clip(lower=0)
            negative_data = scenario_data.clip(upper=0)
            
            x_pos = x + scen_idx * bar_width - (n_scenarios - 1) * bar_width / 2
            
            # Plot positive (causes)
            bottom_pos = np.zeros(n_years)
            for carrier in df_filtered.columns:
                if carrier in positive_data.columns:
                    values = positive_data[carrier].values
                    axes[i].bar(
                        x_pos,
                        values,
                        bar_width,
                        bottom=bottom_pos,
                        color=tech_colors.get(carrier, "gray"),
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.5,
                        hatch=patterns[scen_idx % len(patterns)],
                    )
                    bottom_pos += values
            
            # Collect y values for global limits
            all_y_values.extend(bottom_pos)
            
            # Plot negative (solutions)
            bottom_neg = np.zeros(n_years)
            for carrier in df_filtered.columns:
                if carrier in negative_data.columns:
                    values = negative_data[carrier].values
                    axes[i].bar(
                        x_pos,
                        values,
                        bar_width,
                        bottom=bottom_neg,
                        color=tech_colors.get(carrier, "gray"),
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.5,
                        hatch=patterns[scen_idx % len(patterns)],
                    )
                    bottom_neg += values
            
            # Collect y values for global limits
            all_y_values.extend(bottom_neg)
            
            # Total flexibility needs line
            total_flex = scenario_data.sum(axis=1).values
            all_y_values.extend(total_flex)
            
            for j in range(n_years):
                axes[i].hlines(
                    y=total_flex[j],
                    xmin=x_pos[j] - bar_width/2,
                    xmax=x_pos[j] + bar_width/2,
                    colors="black",
                    linestyles="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
        
        # Formatting
        axes[i].set_title(f"{title_map[gran]} Flexibility Needs", fontsize=14)
        axes[i].set_xlabel("Year", fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(years, rotation=0, fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        
        if i == 0:
            axes[i].set_ylabel("Flexibility Contribution (TWh/a)", fontsize=12)
    
    # Set y-axis limits with margin (15% on each side)
    y_min, y_max = min(all_y_values), max(all_y_values)
    y_range = y_max - y_min
    margin = 0.15
    for ax in axes:
        ax.set_ylim(y_min - y_range * margin, y_max + y_range * margin)
    
    # Create legend for technologies (bottom)
    tech_handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=tech_colors.get(carrier, "gray"),
                   markersize=10)
        for carrier in df_filtered.columns
    ]
    tech_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=0.8)
    )
    tech_labels = list(df_filtered.columns) + ["Total Flexibility Needs"]
    
    fig.legend(
        tech_handles, tech_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(8, len(tech_labels)),
        title="Technologies",
        fontsize=9,
    )
    
    # Create legend for scenarios (top left of first subplot)
    scenario_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.85,
                     edgecolor='white', hatch=patterns[i % len(patterns)])
        for i in range(n_scenarios)
    ]
    
    axes[0].legend(
        scenario_handles, scenarios,
        loc="upper left",
        ncol=1,
        title="Scenarios (bar order)",
        fontsize=9,
        framealpha=0.9,
    )
    
    plt.suptitle("Flexibility Causes by Granularity - Scenario Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    logger.info(f"Saved flexibility causes comparison plot to {output_file}")


def plot_flexibility_provision_scenario_comparison(
    flex_contributions_dict,
    tech_colors,
    output_file,
    granularities=None,
    figsize=(18, 8),
):
    """
    Plot flexibility provision comparison across multiple scenarios.
    Shows stacked bar charts with scenario-specific patterns.
    
    Parameters
    ----------
    flex_contributions_dict : dict
        Dictionary with {scenario: flexibility_df} where flexibility_df has
        MultiIndex (Granularity, Year) and technology columns
    tech_colors : dict
        Dictionary mapping technology names to colors
    output_file : str
        Path to save the plot
    granularities : list, optional
        List of granularities to plot
    figsize : tuple
        Figure size
    """
    
    # Default hatching patterns for scenarios
    patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo']
    
    # Combine all scenarios
    combined_data = {}
    for scenario, df in flex_contributions_dict.items():
        combined_data[scenario] = df
    
    # Get all technologies across all scenarios
    all_techs = set()
    for df in combined_data.values():
        all_techs.update(df.columns)
    all_techs = sorted(all_techs)
    
    # Select granularities
    if granularities is None:
        granularities = ["daily", "weekly", "monthly", "annual"]
    
    first_scenario = list(combined_data.keys())[0]
    available_grans = [
        g for g in granularities
        if g in combined_data[first_scenario].index.get_level_values("Granularity")
    ]
    
    # Setup subplots
    fig, axes = plt.subplots(1, len(available_grans), figsize=figsize, sharey=True)
    if len(available_grans) == 1:
        axes = [axes]
    
    title_map = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "annual": "Annual"}
    
    scenarios = list(combined_data.keys())
    n_scenarios = len(scenarios)
    
    # Find global y limits
    all_y_values = []
    
    # Plot each granularity
    for i, gran in enumerate(available_grans):
        # Get years from first scenario
        first_gran_data = combined_data[first_scenario].xs(gran, level="Granularity")
        years = sorted(first_gran_data.index)
        n_years = len(years)
        
        # Bar positions
        bar_width = 0.8 / n_scenarios
        x = np.arange(n_years)
        
        # Plot each scenario
        for scen_idx, scenario in enumerate(scenarios):
            scenario_data = combined_data[scenario].xs(gran, level="Granularity")
            
            # Separate positive and negative
            positive_data = scenario_data.clip(lower=0)
            negative_data = scenario_data.clip(upper=0)
            
            x_pos = x + scen_idx * bar_width - (n_scenarios - 1) * bar_width / 2
            
            # Plot positive (solutions)
            bottom_pos = np.zeros(n_years)
            for tech in all_techs:
                if tech in positive_data.columns:
                    values = positive_data[tech].values
                    axes[i].bar(
                        x_pos,
                        values,
                        bar_width,
                        bottom=bottom_pos,
                        color=tech_colors.get(tech, "gray"),
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.5,
                        hatch=patterns[scen_idx % len(patterns)],
                    )
                    bottom_pos += values
            
            # Collect y values
            all_y_values.extend(bottom_pos)
            
            # Plot negative
            bottom_neg = np.zeros(n_years)
            for tech in all_techs:
                if tech in negative_data.columns:
                    values = negative_data[tech].values
                    axes[i].bar(
                        x_pos,
                        values,
                        bar_width,
                        bottom=bottom_neg,
                        color=tech_colors.get(tech, "gray"),
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.5,
                        hatch=patterns[scen_idx % len(patterns)],
                    )
                    bottom_neg += values
            
            all_y_values.extend(bottom_neg)
            
            # Total flexibility provision line
            total_flex = scenario_data.sum(axis=1).values
            all_y_values.extend(total_flex)
            
            for j in range(n_years):
                axes[i].hlines(
                    y=total_flex[j],
                    xmin=x_pos[j] - bar_width/2,
                    xmax=x_pos[j] + bar_width/2,
                    colors="black",
                    linestyles="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
        
        # Formatting
        axes[i].set_title(f"{title_map[gran]} Flexibility Provision", fontsize=14)
        axes[i].set_xlabel("Year", fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(years, rotation=0, fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        
        if i == 0:
            axes[i].set_ylabel("Flexibility Contribution (TWh/a)", fontsize=12)
    
    # Set y-axis limits with margin
    y_min, y_max = min(all_y_values), max(all_y_values)
    y_range = y_max - y_min
    margin = 0.15
    for ax in axes:
        ax.set_ylim(y_min - y_range * margin, y_max + y_range * margin)
    
    # Create legend for technologies (bottom)
    tech_handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=tech_colors.get(tech, "gray"),
                   markersize=10)
        for tech in all_techs
    ]
    tech_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=0.8)
    )
    tech_labels = list(all_techs) + ["Total Flexibility Provision"]
    
    fig.legend(
        tech_handles, tech_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(8, len(tech_labels)),
        title="Technologies",
        fontsize=9,
    )
    
    # Create legend for scenarios (top left of first subplot)
    scenario_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.85,
                     edgecolor='white', hatch=patterns[i % len(patterns)])
        for i in range(n_scenarios)
    ]
    
    axes[0].legend(
        scenario_handles, scenarios,
        loc="upper left",
        ncol=1,
        title="Scenarios (bar order)",
        fontsize=9,
        framealpha=0.9,
    )
    
    plt.suptitle("Flexibility Provision by Granularity - Scenario Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    logger.info(f"Saved flexibility provision comparison plot to {output_file}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        project_root = find_project_root()
        os.chdir(project_root)

        snakemake = mock_snakemake(
            "flexibility_plots_scenario_comparison",
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
    flex_needs = {}
    flex_causes_raw = {}
    flex_contributions_clean = {}

    # Load networks
    logger.info("Loading networks...")
    for network_path in snakemake.input.networks:
        path_parts = Path(network_path).parts
        scenario = path_parts[-3]
        filename = Path(network_path).stem
        year = int(filename.split('_')[-1])
        
        if scenario not in networks:
            networks[scenario] = {}
        
        if Path(network_path).exists():
            logger.info(f"  Loading {scenario} network for year {year}...")
            networks[scenario][year] = pypsa.Network(network_path)

    # Load flex_needs
    logger.info("Loading flexibility needs...")
    for i, scenario in enumerate(scenarios):
        flex_needs_path = snakemake.input.flex_needs[i]
        if Path(flex_needs_path).exists():
            flex_needs[scenario] = pd.read_csv(flex_needs_path, index_col=0)
            logger.info(f"  Loaded {scenario}")

    # Load flex_causes_raw
    logger.info("Loading flexibility causes (raw)...")
    for i, scenario in enumerate(scenarios):
        flex_causes_path = snakemake.input.flex_causes_raw[i]
        if Path(flex_causes_path).exists():
            flex_causes_raw[scenario] = pd.read_pickle(flex_causes_path)
            logger.info(f"  Loaded {scenario}")

    # Load flex_contributions_clean
    logger.info("Loading flexibility contributions (clean)...")
    for i, scenario in enumerate(scenarios):
        flex_contrib_path = snakemake.input.flex_contributions_clean[i]
        if Path(flex_contrib_path).exists():
            flex_contributions_clean[scenario] = pd.read_csv(flex_contrib_path, index_col=[0, 1])
            logger.info(f"  Loaded {scenario}")

    # Create output directory
    output_dir = Path(snakemake.params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("All data loaded successfully!")
    logger.info(f"Loaded {len(networks)} scenarios with networks")
    logger.info(f"Scenarios: {list(flex_needs.keys())}")

    ####### PLOTTING #########

    # Flex needs   
    logger.info("Plotting flexibility needs comparison...")
    plot_flex_needs_comparison(
        flex_needs,
        snakemake.output.flex_needs_comparison,
        colormaps=['Blues', 'Oranges', 'Greens'],
    )

    # Flex causes
    plot_flexibility_causes_scenario_comparison(
        flex_causes_raw, 
        tech_colors,
        output_dir / "flex_causes_scenario_comparison.png",
    )

    # Flex provision
    plot_flexibility_provision_scenario_comparison(
        flex_contributions_clean,  # Your dict of {scenario: flexibility_df}
        tech_colors,
        output_dir / "flex_provision_scenario_comparison.png",
    )

    # System cost comparison
