import logging
import sys

import pandas as pd
import pypsa
from xarray import DataArray

from scripts.prepare_sector_network import determine_emission_sectors

logger = logging.getLogger(__name__)


def add_capacity_limits(n, investment_year, limits_capacity, sense="maximum"):
    for c in n.iterate_components(limits_capacity):
        logger.info(f"Adding {sense} constraints for {c.list_name}")

        attr = "e" if c.name == "Store" else "p"
        units = "MWh or tCO2" if c.name == "Store" else "MW"

        for carrier in limits_capacity[c.name]:
            for ct in limits_capacity[c.name][carrier]:
                if investment_year not in limits_capacity[c.name][carrier][ct].keys():
                    continue

                limit = 1e3 * limits_capacity[c.name][carrier][ct][investment_year]

                logger.info(
                    f"Adding constraint on {c.name} {carrier} capacity in {ct} to be {sense} {limit} {units}"
                )

                valid_components = (
                    (c.df.index.str[:2] == ct)
                    & (c.df.carrier.str[: len(carrier)] == carrier)
                    & ~c.df.carrier.str.contains("thermal")
                )  # exclude solar thermal

                existing_index = c.df.index[
                    valid_components & ~c.df[attr + "_nom_extendable"]
                ]
                extendable_index = c.df.index[
                    valid_components & c.df[attr + "_nom_extendable"]
                ]

                existing_capacity = c.df.loc[existing_index, attr + "_nom"].sum()

                logger.info(
                    f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} {units}"
                )

                nom = n.model[c.name + "-" + attr + "_nom"].loc[extendable_index]

                lhs = nom.sum()

                cname = f"capacity_{sense}-{ct}-{c.name}-{carrier.replace(' ', '-')}"

                if cname in n.global_constraints.index:
                    logger.warning(
                        f"Global constraint {cname} already exists. Dropping and adding it again."
                    )
                    n.global_constraints.drop(cname, inplace=True)

                rhs = limit - existing_capacity

                if sense == "maximum":
                    if rhs <= 0:
                        logger.warning(
                            f"Existing capacity in {ct} for carrier {carrier} already exceeds the limit of {limit} MW. Limiting capacity expansion for this investment period to 0."
                        )
                        rhs = 0

                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense="<=",
                        type="",
                        carrier_attribute="",
                    )

                elif sense == "minimum":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense=">=",
                        type="",
                        carrier_attribute="",
                    )
                else:
                    logger.error("sense {sense} not recognised")
                    sys.exit()


def add_power_limits(n, investment_year, limits_power_max):
    """
    " Restricts the maximum inflow/outflow of electricity from/to a country.
    """
    for ct in limits_power_max:
        if investment_year not in limits_power_max[ct].keys():
            continue

        limit = 1e3 * limits_power_max[ct][investment_year] / 10

        logger.info(
            f"Adding constraint on electricity import/export from/to {ct} to be < {limit} MW"
        )
        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        # iterate over snapshots - otherwise exporting of postnetwork fails since
        # the constraints are time dependent
        for t in n.snapshots:
            incoming_line_p = n.model["Line-s"].loc[t, incoming_line]
            outgoing_line_p = n.model["Line-s"].loc[t, outgoing_line]
            incoming_link_p = n.model["Link-p"].loc[t, incoming_link]
            outgoing_link_p = n.model["Link-p"].loc[t, outgoing_link]

            lhs = (
                incoming_link_p.sum()
                - outgoing_link_p.sum()
                + incoming_line_p.sum()
                - outgoing_line_p.sum()
            ) / 10
            # divide by 10 to avoid numerical issues

            cname_upper = f"Power-import-limit-{ct}-{t}"
            cname_lower = f"Power-export-limit-{ct}-{t}"

            n.model.add_constraints(lhs <= limit, name=cname_upper)
            n.model.add_constraints(lhs >= -limit, name=cname_lower)

            # not adding to network as the shadow prices are not needed


def h2_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_import"]:
        limit = limits_volume_max["h2_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 imports in {ct} to {limit / 1e6} TWh/a")
        pipeline_carrier = [
            "H2 pipeline",
            "H2 pipeline (Kernnetz)",
            "H2 pipeline retrofitted",
        ]
        incoming = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_p = (
            n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
        ).sum()
        outgoing_p = (
            n.model["Link-p"].loc[:, outgoing] * n.snapshot_weightings.generators
        ).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )

        logger.info("Adding H2 export ban")

        cname = f"H2_export_ban-{ct}"

        n.model.add_constraints(lhs >= 0, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=0,
            sense=">=",
            type="",
            carrier_attribute="",
        )


def h2_production_limits(n, investment_year, limits_volume_min, limits_volume_max):
    for ct in limits_volume_max["electrolysis"]:
        if ct not in limits_volume_min["electrolysis"]:
            logger.warning(
                f"no lower limit for H2 electrolysis in {ct} assuming 0 TWh/a"
            )
            limit_lower = 0
        else:
            limit_lower = limits_volume_min["electrolysis"][ct][investment_year] * 1e6

        limit_upper = limits_volume_max["electrolysis"][ct][investment_year] * 1e6

        logger.info(
            f"limiting H2 electrolysis in DE between {limit_lower / 1e6} and {limit_upper / 1e6} TWh/a"
        )

        production = n.links[
            (n.links.carrier == "H2 Electrolysis") & (n.links.bus0.str.contains(ct))
        ].index
        efficiency = n.links.loc[production, "efficiency"]

        lhs = (
            n.model["Link-p"].loc[:, production]
            * n.snapshot_weightings.generators
            * efficiency
        ).sum()

        cname_upper = f"H2_production_limit_upper-{ct}"
        cname_lower = f"H2_production_limit_lower-{ct}"

        n.model.add_constraints(
            lhs <= limit_upper, name=f"GlobalConstraint-{cname_upper}"
        )

        n.model.add_constraints(
            lhs >= limit_lower, name=f"GlobalConstraint-{cname_lower}"
        )

        if cname_upper not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_upper,
                constant=limit_upper,
                sense="<=",
                type="",
                carrier_attribute="",
            )
        if cname_lower not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_lower,
                constant=limit_lower,
                sense=">=",
                type="",
                carrier_attribute="",
            )


def electricity_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["electricity_import"]:
        limit = limits_volume_max["electricity_import"][ct][investment_year] * 1e6

        if limit < 0:
            limit *= n.snapshot_weightings.generators.sum() / 8760

        logger.info(f"limiting electricity imports in {ct} to {limit / 1e6} TWh/a")

        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_line_p = (
            n.model["Line-s"].loc[:, incoming_line] * n.snapshot_weightings.generators
        ).sum()
        outgoing_line_p = (
            n.model["Line-s"].loc[:, outgoing_line] * n.snapshot_weightings.generators
        ).sum()

        incoming_link_p = (
            n.model["Link-p"].loc[:, incoming_link] * n.snapshot_weightings.generators
        ).sum()
        outgoing_link_p = (
            n.model["Link-p"].loc[:, outgoing_link] * n.snapshot_weightings.generators
        ).sum()

        lhs = (incoming_link_p - outgoing_link_p) + (incoming_line_p - outgoing_line_p)

        cname = f"Electricity_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def add_national_co2_budgets(n, snakemake, national_co2_budgets, investment_year):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_national'.

    Parameters
    ----------
    n : pypsa.Network
    snakemake : snakemake.io.Snakemake
    national_co2_budgets : dict
    investment_year : int

    """
    logger.info("Adding national CO2 budgets")
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    sectors = determine_emission_sectors(n.config["sector"])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in national_co2_budgets:
        if ct != "DE":
            logger.error(
                f"CO2 budget for countries other than `DE` is not yet supported. Found country {ct}. Please check the config file."
            )

        limit = co2_total_totals[ct] * national_co2_budgets[ct][investment_year]
        logger.info(
            f"Limiting emissions in country {ct} to {national_co2_budgets[ct][investment_year]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:
            links = n.links.index[
                (n.links.index.str[:2] == ct)
                & (n.links[f"bus{port}"] == "co2 atmosphere")
                & (
                    n.links.carrier != "kerosene for aviation"
                )  # first exclude aviation to multiply it with a domestic factor later
            ]

            logger.info(
                f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links, 'carrier'].unique()}"
            )

            if port == "0":
                efficiency = -1.0
            elif port == "1":
                efficiency = n.links.loc[links, "efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append(
                (
                    n.model["Link-p"].loc[:, links]
                    * efficiency
                    * n.snapshot_weightings.generators
                ).sum()
            )

        # Aviation demand
        energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])
        domestic_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total domestic aviation"
        ]
        international_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total international aviation"
        ]
        domestic_factor = domestic_aviation / (
            domestic_aviation + international_aviation
        )
        aviation_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "kerosene for aviation")
        ]
        lhs.append
        (
            n.model["Link-p"].loc[:, aviation_links.index]
            * aviation_links.efficiency2
            * n.snapshot_weightings.generators
        ).sum() * domestic_factor
        logger.info(
            f"Adding domestic aviation emissions for {ct} with a factor of {domestic_factor}"
        )

        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == f"EU renewable oil -> {ct} oil"]
        outgoing_oil = n.links.index[n.links.index == f"{ct} renewable oil -> EU oil"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )
        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )

        incoming_methanol = n.links.index[
            n.links.index == f"EU methanol -> {ct} methanol"
        ]
        outgoing_methanol = n.links.index[
            n.links.index == f"{ct} methanol -> EU methanol"
        ]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        # Methane
        incoming_CH4 = n.links.index[n.links.index == f"EU renewable gas -> {ct} gas"]
        outgoing_CH4 = n.links.index[n.links.index == f"{ct} renewable gas -> EU gas"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def force_boiler_profiles_existing_per_load(n):
    """
    This scales the boiler dispatch to the load profile with a factor common to
    all boilers at load.
    """

    logger.info("Forcing boiler profiles for existing ones")

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    boiler_load_index = pd.Index(boiler_loads.unique())
    boiler_load_index.name = "Load"

    # per load scaling factor
    n.model.add_variables(coords=[boiler_load_index], name="Load-profile_factor")

    # clumsy indicator matrix to map boilers to loads
    df = pd.DataFrame(index=boiler_load_index, columns=decentral_boilers, data=0.0)
    for k, v in boiler_loads.items():
        df.loc[v, k] = 1.0

    lhs = n.model["Link-p"].loc[:, decentral_boilers] - (
        boiler_profiles * DataArray(df) * n.model["Load-profile_factor"]
    ).sum("Load")

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.loads["profile_factor_opt"] = 0.0


def force_boiler_profiles_existing_per_boiler(n):
    """
    This scales each boiler dispatch to be proportional to the load profile.
    """

    logger.info(
        "Forcing each existing boiler dispatch to be proportional to the load profile"
    )

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    # will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (
        (1, n.model["Link-p"].loc[:, decentral_boilers]),
        (
            -boiler_profiles,
            n.model["Link-fixed_profile_scaling"],
        ),
    )

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.0


def add_h2_derivate_limit(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_derivate_import"]:
        limit = limits_volume_max["h2_derivate_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 derivate imports in {ct} to {limit / 1e6} TWh/a")

        incoming = n.links.loc[
            [
                "EU renewable oil -> DE oil",
                "EU methanol -> DE methanol",
                "EU renewable gas -> DE gas",
            ]
        ].index
        outgoing = n.links.loc[
            [
                "DE renewable oil -> EU oil",
                "DE methanol -> EU methanol",
                "DE renewable gas -> EU gas",
            ]
        ].index

        carrier_idx_dict = {
            # Every carrier should respect the limit individually
            "renewable_oil": 0,
            "methanol": 1,
            "renewable_gas": 2,
            # Exports of one carrier should not compensate for imports of another carrier
            "H2_derivate_oil_meoh": [0, 1],
            "H2_derivate_oil_gas": [0, 2],
            "H2_derivate_meoh_gas": [1, 2],
            # The sum of all carriers should respect the limit
            "H2_derivate_oil_meoh_gas": [0, 1, 2],
        }
        for carrier, idx in carrier_idx_dict.items():
            cname = f"{carrier}_import_limit-{ct}"

            incoming_p = (
                n.model["Link-p"].loc[:, incoming[idx]]
                * n.snapshot_weightings.generators
            ).sum()
            outgoing_p = (
                n.model["Link-p"].loc[:, outgoing[idx]]
                * n.snapshot_weightings.generators
            ).sum()

            lhs = incoming_p - outgoing_p

            n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

            if cname in n.global_constraints.index:
                logger.warning(
                    f"Global constraint {cname} already exists. Dropping and adding it again."
                )
                n.global_constraints.drop(cname, inplace=True)

            n.add(
                "GlobalConstraint",
                cname,
                constant=limit,
                sense="<=",
                type="",
                carrier_attribute="",
            )

    # Export bans on efuels are implemented in modify_prenetwork by restricting p_max_pu of the DE -> EU links


def adapt_nuclear_output(n):
    logger.info(
        "limiting german electricity generation from nuclear to 2020 value of 61 TWh"
    )
    limit = 61e6

    nuclear_de_index = n.links.index[
        (n.links.carrier == "nuclear") & (n.links.index.str[:2] == "DE")
    ]

    nuclear_gen = (
        n.model["Link-p"].loc[:, nuclear_de_index]
        * n.links.loc[nuclear_de_index, "efficiency"]
        * n.snapshot_weightings.generators
    ).sum()

    lhs = nuclear_gen

    cname = "Nuclear_generation_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if cname in n.global_constraints.index:
        logger.warning(
            f"Global constraint {cname} already exists. Dropping and adding it again."
        )
        n.global_constraints.drop(cname, inplace=True)

    n.add(
        "GlobalConstraint",
        cname,
        constant=limit,
        sense="<=",
        type="",
        carrier_attribute="",
    )


def add_industry_dsm_cycling_constraint(n, industry_dsm):
    """
    Add constraint to ensure DSM debt stores are empty every X hours.

    Works with any temporal resolution, including coarse resolutions like 365H.

    Parameters
    ----------
    n : pypsa.Network
        The energy system network with DSM components already added.
    industry_dsm : dict
        Configuration dictionary containing:
        - compensate_hours: int, hours between forced zero debt (e.g. 24 or 48)
    """
    compensate_hours = industry_dsm.get("compensate_hours", 24)

    logger.info(
        f"Adding DSM cycling constraint: debt must be zero every {compensate_hours} hours"
    )

    # Find all DSM debt stores
    dsm_stores = n.stores.index[n.stores.carrier == "industry DSM"]

    if dsm_stores.empty:
        logger.warning("No DSM debt stores found. Skipping cycling constraint.")
        return

    # Convert snapshots to hours from start
    start_time = n.snapshots[0]
    snapshot_hours = [
        (snap - start_time).total_seconds() / 3600 for snap in n.snapshots
    ]

    # Find snapshots that are at multiples of compensate_hours
    # Allow some tolerance for floating point comparison
    tolerance = 0.1  # hours
    cycle_snapshots = []

    for i, hours in enumerate(snapshot_hours):
        # Check if this snapshot is at a multiple of compensate_hours
        remainder = hours % compensate_hours
        # Check both remainder near 0 or near compensate_hours (wrapping)
        if remainder < tolerance or (compensate_hours - remainder) < tolerance:
            cycle_snapshots.append(n.snapshots[i])

    # Always include the last snapshot to ensure debt is cleared at end
    if n.snapshots[-1] not in cycle_snapshots:
        cycle_snapshots.append(n.snapshots[-1])

    if not cycle_snapshots:
        logger.warning(
            f"No snapshots found at {compensate_hours}h intervals. "
            f"Temporal resolution may be too coarse. Adding constraint only at last snapshot."
        )
        cycle_snapshots = [n.snapshots[-1]]

    # Calculate average snapshot duration for info
    if len(n.snapshots) > 1:
        avg_duration = sum(
            (n.snapshots[i + 1] - n.snapshots[i]).total_seconds() / 3600
            for i in range(len(n.snapshots) - 1)
        ) / (len(n.snapshots) - 1)
        logger.info(f"Average snapshot duration: {avg_duration:.2f} hours")

    logger.info(
        f"DSM debt must be zero at {len(cycle_snapshots)} snapshots "
        f"(approximately every {compensate_hours} hours)"
    )

    # Add constraints for each store at each cycle point
    constraint_count = 0
    for store in dsm_stores:
        for snapshot in cycle_snapshots:
            cname = f"DSM_cycling-{store}-{snapshot}"

            # Store state of charge must be zero at this snapshot
            lhs = n.model["Store-e"].loc[snapshot, store]

            n.model.add_constraints(lhs == 0, name=cname)
            constraint_count += 1

    # not adding to network as the shadow prices are not needed

    logger.info(
        f"Added {constraint_count} DSM cycling constraints across {len(dsm_stores)} stores"
    )


def force_pth_profiles_decentral_rural(n, flexibility_margin=0.0):
    """
    Constrains PtH asset dispatch to follow load profile with optional flexibility margin.

    Parameters
    ----------
    n : pypsa.Network
        The network object
    flexibility_margin : float, default 0.0
        Allowed deviation from profile as a fraction (e.g., 0.1 for ±10%).
        - 0.0: Strict equality constraint (no flexibility)
        - >0.0: Assets can operate within [profile*(1-margin), profile*(1+margin)]

    Applies to heat pumps and resistive heaters in rural and urban decentral areas.
    """
    logger.info(
        f"Constraining PtH dispatch to load profile "
        f"(flexibility margin: {flexibility_margin * 100:.1f}%)"
    )

    # Filter for PtH assets in rural and decentral areas
    pth_links = n.links.index[
        (
            (
                n.links.carrier.str.contains("rural")
                | n.links.carrier.str.contains("decentral")
            )
            & (
                n.links.carrier.str.contains("heat pump")
                | n.links.carrier.str.contains("resistive heater")
            )
        )
        & ~n.links.carrier.str.contains("urban central")
    ]

    if pth_links.empty:
        logger.warning("No PtH links found matching criteria")
        return

    # Get the heat buses these PtH assets supply
    pth_loads = n.links.loc[pth_links, "bus1"]
    pth_loads = pth_loads[pth_loads.isin(n.loads_t.p_set.columns)]
    pth_links = pth_loads.index

    if pth_links.empty:
        logger.warning("No valid heat loads found for PtH links")
        return

    # Create normalized load profiles (per-unit, 0-1 range)
    pth_profiles_pu = n.loads_t.p_set[pth_loads].div(
        n.loads_t.p_set[pth_loads].max(), axis=1
    )
    pth_profiles_pu.columns = pth_links

    # Separate extendable and non-extendable links
    extendable = n.links.loc[pth_links, "p_nom_extendable"]
    pth_links_ext = pth_links[extendable]
    pth_links_fixed = pth_links[~extendable]

    if flexibility_margin == 0.0:
        # Strict equality constraint

        # For extendable links: Link-p = profile_pu × Link-p_nom (variable)
        if not pth_links_ext.empty:
            pth_profiles_pu_da_ext = DataArray(pth_profiles_pu[pth_links_ext])

            # Align dimensions by selecting Link-p_nom with matching indices
            p_nom_ext = n.model["Link-p_nom"].loc[pth_links_ext]

            # Rename the dimension to match Link-p dimension
            p_nom_ext = p_nom_ext.rename({"Link-ext": "Link"})

            lhs_ext = (
                (1, n.model["Link-p"].loc[:, pth_links_ext]),
                (-pth_profiles_pu_da_ext, p_nom_ext),
            )

            n.model.add_constraints(lhs_ext, "=", 0, "Link-pth_profile_extendable")
            logger.info(
                f"Applied strict equality constraint to {len(pth_links_ext)} extendable PtH assets"
            )

        # For non-extendable links: Link-p = profile_pu × p_nom (fixed)
        if not pth_links_fixed.empty:
            pth_profiles_fixed = DataArray(
                pth_profiles_pu[pth_links_fixed].multiply(
                    n.links.loc[pth_links_fixed, "p_nom"], axis=1
                )
            )

            lhs_fixed = (
                (1, n.model["Link-p"].loc[:, pth_links_fixed]),
            )  # Note: tuple of tuple

            n.model.add_constraints(
                lhs_fixed, "=", pth_profiles_fixed, "Link-pth_profile_fixed"
            )
            logger.info(
                f"Applied strict equality constraint to {len(pth_links_fixed)} non-extendable PtH assets"
            )

    else:
        # Flexible band: profile × (1±margin)
        # Initialize if needed
        if not hasattr(n, "links_t"):
            n.links_t = pypsa.descriptors.Dict()

        if not hasattr(n.links_t, "p_min_pu") or n.links_t.p_min_pu.empty:
            n.links_t.p_min_pu = pd.DataFrame(
                0.0, index=n.snapshots, columns=n.links.index
            )
        else:
            n.links_t.p_min_pu = n.links_t.p_min_pu.reindex(
                columns=n.links.index, fill_value=0.0
            )

        if not hasattr(n.links_t, "p_max_pu") or n.links_t.p_max_pu.empty:
            n.links_t.p_max_pu = pd.DataFrame(
                1.0, index=n.snapshots, columns=n.links.index
            )
        else:
            n.links_t.p_max_pu = n.links_t.p_max_pu.reindex(
                columns=n.links.index, fill_value=1.0
            )

        # Set min and max bounds (works for both extendable and non-extendable)
        n.links_t.p_min_pu[pth_links] = pth_profiles_pu * (1 - flexibility_margin)
        n.links_t.p_max_pu[pth_links] = pth_profiles_pu * (1 + flexibility_margin)

        # Ensure bounds don't go below 0 or above 1
        n.links_t.p_min_pu[pth_links] = n.links_t.p_min_pu[pth_links].clip(lower=0.0)
        n.links_t.p_max_pu[pth_links] = n.links_t.p_max_pu[pth_links].clip(upper=1.0)

        logger.info(
            f"Applied ±{flexibility_margin * 100:.1f}% flexibility band to "
            f"{len(pth_links)} PtH assets ({len(pth_links_ext)} extendable, "
            f"{len(pth_links_fixed)} fixed)"
        )


def additional_functionality(n, snapshots, snakemake):
    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    constraints = snakemake.params.solving["constraints"]

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_min"], "minimum"
    )

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_max"], "maximum"
    )

    add_power_limits(n, investment_year, constraints["limits_power_max"])

    limits_volume_max = constraints.get("limits_volume_max", None)
    limits_volume_min = constraints.get("limits_volume_min", None)

    if snakemake.wildcards.clusters != "1":
    
        if (
            "h2_import" in constraints["limits_volume_max"]
            and any(
                investment_year in year_limits
                for year_limits in limits_volume_max["h2_import"].values()
            )
        ):
            h2_import_limits(n, investment_year, constraints["limits_volume_max"])


        if (
            "electricity_import" in constraints["limits_volume_max"]
            and any(
                investment_year in year_limits
                for year_limits in limits_volume_max["electricity_import"].values()
            )
        ):
            electricity_import_limits(n, investment_year, constraints["limits_volume_max"])

    
    if limits_volume_max is not None and limits_volume_min is not None:
        if (investment_year >= 2025) & ("electrolysis" in constraints["limits_volume_max"]) & ("electrolysis" in constraints["limits_volume_min"]):
            h2_production_limits(
                n,
                investment_year,
                constraints["limits_volume_min"],
                constraints["limits_volume_max"],
            )

        if "h2_derivate_import" in constraints["limits_volume_max"]:
            add_h2_derivate_limit(n, investment_year, constraints["limits_volume_max"])

    # force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    if isinstance(constraints["co2_budget_national"], dict):
        add_national_co2_budgets(
            n,
            snakemake,
            constraints["co2_budget_national"],
            investment_year,
        )
    else:
        logger.warning("No national CO2 budget specified!")

    if investment_year == 2020:
        adapt_nuclear_output(n)

    # Flexibility implementations

    if (snakemake.params.industry_dsm["enable"]) & (
        investment_year in snakemake.params.industry_dsm.keys()
    ):
        add_industry_dsm_cycling_constraint(
            n, snakemake.params.industry_dsm[investment_year]
        )

    if snakemake.params.solving.get("force_pth_profiles_decentral_rural", False):
        force_pth_profiles_decentral_rural(
            n,
            snakemake.params.solving.get(
                "force_pth_profiles_decentral_rural_margin", 0.0
            ),
        )
