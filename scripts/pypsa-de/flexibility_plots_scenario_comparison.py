import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy
import pypsa
import pickle
import matplotlib.pyplot as plt
from flexibility_utils import tech_colors, year_colors_gradient, tech_groups
from flexibility_analysis import aggregate_by_keywords
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        project_root = os.path.dirname(os.getcwd())  # Go up one level from 'scripts'
        project_root = "/home/julian-geis/repos/pypsa-de-flex"  # Set your project root here
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

