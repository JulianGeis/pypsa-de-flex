from pathlib import Path
from numpy import isclose
import matplotlib.pyplot as plt


NON_DISPATCHABLE_SUPPLY_CARRIERS = [
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "solar",
    "solar-hsat",
    "solar rooftop",
    "ror",
]

NON_DISPATCHABLE_DEMAND_CARRIERS = [
    "electricity",
    "agriculture electricity",
    "industry electricity",
    "agriculture machinery electric",
    "land transport EV",
    'rural air heat pump',
    'rural ground heat pump',
    'urban decentral air heat pump',
    "rural resistive heater", 
    "urban decentral resistive heater",
]


tech_colors = {
    "AC": "#70af1d",
    "DC": "#8a1caf",
    "solar": "#f9d002",
    "onwind": "#235ebc",
    "hydro": "#298c81",
    "offwind-dc": "#74c6f2",
    "solar-hsat": "#fdb915",
    "offwind-ac": "#6895dd",
    "ror": "#3dbfb0",
    "PHS": "#51dbcc",
    "lignite": "#826837",
    "lignite (+CHP)": "#826837",
    "coal": "black",
    "coal (+CHP)": "black",
    "oil": "#c9c9c9",
    "Oil": "black",
    "uranium": "#ff8c00",
    "none": "",
    "co2": "#f29dae",
    "co2 stored": "#f2385a",
    "co2 sequestered": "#f2682f",
    "gas": "#e0986c",
    "H2": "#bf13a0",
    "battery": "fuchsia",
    "EV battery": "#baf238",
    "urban central heat": "#d15959",
    "urban central water tanks": "#e9977d",
    "urban central solar thermal": "#d7b24c",
    "biogas": "#e3d37d",
    "solid biomass": "#baa741",
    "biomass (+CHP)": "#baa741",
    "methanol": "#FF7B00",
    "home battery": "#80c944",
    "unsustainable biogas": "",
    "unsustainable solid biomass": "#998622",
    "unsustainable bioliquids": "#32CD32",
    "H2 Store": "#bf13a0",
    "gas for industry CC": "#692e0a",
    "urban central air heat pump": "#6cfb6b",
    "battery discharger": "palegreen",
    "urban central resistive heater": "#8cdf85",
    "biomass to liquid CC": "#32DDD2",
    "urban central solid biomass CHP": "#9d9042",
    "battery charger": "#88a75b",
    "home battery discharger": "#3c5221",
    "H2 pipeline retrofitted": "#ba99b5",
    "DAC": "#ff5270",
    "solid biomass for industry CC": "#47411c",
    "land transport oil": "#afafaf",
    "BEV charger": "#baf238",
    "BEV charging": "#baf238",
    "industry methanol": "#468c8b",
    "shipping oil": "#808080",
    "OCGT": "#e0986c",
    "urban central gas boiler": "#b0904f",
    "gas compressing": "#e05b09",
    "kerosene for aviation": "#a1ffe6",
    "HVC to air": "k",
    "SMR": "#870c71",
    "home battery charger": "#5e8032",
    "coal for industry": "#343434",
    "electricity distribution grid": "#97ad8c",
    "urban central water tanks discharger": "#b9816e",
    "urban central solid biomass CHP CC": "#6c5d28",
    "methanolisation": "#00FFBF",
    "biogas to gas": "#e36311",
    "waste CHP CC": "#e3d3ff",
    "Fischer-Tropsch": "#25c49a",
    "process emissions": "#222222",
    "biomass to liquid": "#32CD32",
    "biogas to gas CC": "#e51245",
    "waste CHP": "#e3d37d",
    "H2 pipeline": "#f081dc",
    "Sabatier": "#9850ad",
    "naphtha for industry": "#57ebc4",
    "agriculture machinery oil": "#949494",
    "H2 Fuel Cell": "#c251ae",
    "urban central gas CHP": "#8d5e56",
    "oil refining": "#a6a6a6",
    "solid biomass for industry": "#7a6d26",
    "SMR CC": "#4f1745",
    "H2 Electrolysis": "#ff29d9",
    "gas for industry": "#853403",
    "urban central gas CHP CC": "#6e4e4c",
    "urban central water tanks charger": "#b57a67",
    "process emissions CC": "#000000",
    "urban central heat vent": "#a74747",
    "oil primary": "#d2d2d2",
    "gas primary": "#e05b09",
    "solar rooftop": "#ffea80",
    "land transport EV": "#baf238",
    "industry electricity": "violet",
    "industrial demand": "violet",
    "low-temperature heat for industry": "#8f2727",
    "H2 for industry": "#f073da",
    "agriculture heat": "#d9a5a5",
    "land transport fuel cell": "#6b3161",
    "electricity": "#110d63",
    "demand": "#110d63",
    "agriculture electricity": "#494778",
    "low voltage": "#97ad8c",
    "non-sequestered HVC": "",
    "rural air heat pump": "#5af95d",
    "rural biomass boiler": "#a1a066",
    "rural gas boiler": "#d4722e",
    "rural ground heat pump": "#48f74f",
    "rural heat": "#ff7c7c",
    "rural resistive heater": "#a5ed9d",
    "rural solar thermal": "#f1c069",
    "rural water tanks": "#f7b7a3",
    "rural water tanks charger": "#b3abb0",
    "rural water tanks discharger": "#ba9685",
    "urban decentral air heat pump": "#5af95d",
    "urban decentral biomass boiler": "#b0b87b",
    "urban decentral gas boiler": "#ba8947",
    "urban decentral heat": "#a33c3c",
    "urban decentral resistive heater": "#98e991",
    "urban decentral solar thermal": "#e5bc5a",
    "urban decentral water tanks": "#f2b2a3",
    "urban decentral water tanks charger": "#b3becc",
    "urban decentral water tanks discharger": "#baac9e",
    "urban central lignite CHP": "#f7a889",
    "urban central oil CHP": "#e26952",
    "urban decentral oil boiler": "",
    "urban central coal CHP": "#b40426",
    "rural oil boiler": "",
    "CCGT": "purple",
    "gas (+CHP)": "purple",
    "nuclear": "lime",
    "renewable oil": "darkviolet",
    "renewable gas": "gold",
    "Other": "gray",
    "H2 OCGT": "#3b4cc0",
    "H2 pipeline (Kernnetz)": "#6788ee",
    "H2 retrofit OCGT": "#9abbff",
    "urban central H2 CHP": "#c9d7f0",
    "urban central H2 retrofit CHP": "#edd1c2",
    "interconnectors": "darkorange",
    "coal CHP": "#b40426",
    "H2 CHP": "pink",
    "H2 (+CHP)": "pink",
    "biomass CHP": "#9d9042",
    "Fuel Cell": "#c251ae",
    "import": "orange",
    "export": "purple",
    "interconnectors supply": "orange",
    "interconnectors demand": "purple",
    "PHS charging": "darkgreen",
    "PHS discharging": "darkgreen",
    "heat pump": "red",
    "heat pump (central)": "red",   
    "heat pump (decentral)": "firebrick",   
    "resistive heater": "khaki",
    "resistive heater (central)": "khaki",
    "resistive heater (decentral)": "darkkhaki",
    "Import/Export": "dimgrey",
    "Sonstige": "gainsboro",
    "industry DSM": "navy",
    "Industry DSM": "navy",
    "industry DSM ramp down": "mediumslateblue",
    "industry DSM compensate": "cornflowerblue",
    "Onshore wind": "#235ebc",
    "Offshore wind": "#6788ee",
    "Solar": "#ffea80",
    "Gas": "#e0986c",
    "Gas CHP": "#e0986c",
    "Pumped storage": "#51dbcc",
    "Battery": "palegreen",
    "Power-to-heat": "crimson",
    "Power-to-heat (central)": "crimson",
    "Power-to-heat (decentral)": "crimson",
    "Electrolysis": "#ff29d9",
    "Iron-air battery": "#c9954d",
    "Others": "grey",
    "V2G": "tomato",
    "Vehicle-to-grid": "tomato",
    "iron-air battery": "#f5e6b3",
    "iron-air battery storage": "#daa520",
    "iron-air battery charger": "#c9954d",
    "iron-air battery discharger": "#8b6f47",
    "electricity distribution grid losses": "#97ad8c",
    "agriculture machinery electricity": "#6b3161",
    "eFuels": "seagreen",
    "Renewable Gas": "gold",
    "Renewable Oil": "darkviolet",
    "Methanol": "#FF7B00",
    "Coal incl. CHP": "#826837",
}

sector_colors = {
    'Electricity': '#110d63',
    'Heat': '#d15959',
    'H2': '#bf13a0',
    'Fuels': '#1abc9c',
    'Gas': '#e0986c',
    'Biomass': '#baa741',
    'Other': 'lightgrey'
    }

year_colors_gradient = {
    2020: "#e4c1f9",  # Pale Lavender
    2025: "#c191c9",  # Soft Lilac
    2030: "#9c89b8",  # Lavender Purple
    2035: "#7b68ac",  # Muted Purple
    2040: "#5e4fa2",  # Deep Violet
    2045: "#3c096c",  # Intense Purple
    2050: "#240046",  # Dark Purple
}

# Technology grouping dictionary
tech_groups = {
    "gas (+CHP)": ["gas CHP", "OCGT", "CCGT"],
    "coal (+CHP)": ["coal", "lignite"],
    "oil (+CHP)": ["oil"],
    "waste (+CHP)": ["waste"],
    "H2 (+CHP)": ["H2 retrofit", "H2 CCGT", "H2 OCGT", "H2 CHP"],
    "biomass (+CHP)": ["biomass"],
    "heat pump (central)": ["urban central air heat pump"],
    "resistive heater (central)": ["urban central resistive heater"],
    "heat pump (decentral)": ["rural air heat pump","rural ground heat pump","urban decentral air heat pump"],
    "resistive heater (decentral)": ["rural resistive heater", "urban decentral resistive heater"],
    "industry DSM": ["industry DSM"],
    "interconnectors": [
        "AC",
        "DC",
        "import",
        "export",
        "electricity distribution grid losses",
    ],
    "V2G": ["V2G", "V2G charging", "V2G discharging losses"],
    "battery charger": ["battery charger", "home battery charger"],
    "battery discharger": ["battery discharger", "home battery discharger"],
}

def find_project_root():
    """Find project root by looking for .git or Snakefile."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "Snakefile").exists():
            return parent
    return current.parent


# Scenario abbreviations dictionary
scenario_abbrev = {
    # Main scenarios
    "LowFlex": "LF",
    "LowBattery": "LB",
    "Base": "BA",
    "HighFlex": "HF",
    
    # LowFlex variations
    "LowFlexEnergyAndPowerCapRestrict" : "LF_EPC",
    "LowFlex75": "LF75",
    "LowFlex75_Electrolysis_pminpu_0.5_Other_pminpu_1": "LF75_E5O1",
    "LowFlex75_Electrolysis_pminpu_0.3_Other_pminpu_1": "LF75_E3O1",
    "LowFlex75_excludeBattery_Electrolysis_pminpu_0.5_Other_pminpu_1": "LF75_XBE5",
    "LowFlex75_excludeBattery_Electrolysis_pminpu_0.3_Other_pminpu_1": "LF75_XBE3",
    "LowFlex75_excludeHeatStorage": "LF75_XHS",
    "LowFlex75_exclude_pminpu": "LF75_XP",
    "LowFlex50": "LF50",
    "LowFlex50_pminpu0.8": "LF50_P8",
    "LowFlex50_pminpu0.5": "LF50_P5",
    "LowFlex50_pminpu0.3": "LF50_P3",
    "LowFlex50_exclude_pminpu": "LF50_XP",
    "LowFlex50_Electrolysis_pminpu_0.5_Other_pminpu_1": "LF50_E5O1",
    "LowFlex50_Electrolysis_pminpu_0.3_Other_pminpu_1": "LF50_E3O1",
    "LowFlex50HeatPMINPU": "LF50_HP",
    "LowFlex50HeatMinStore": "LF50_HMS",

    # LowFlex sensitivities
    "LowFlexElectrolysis_pminpu_0.5": "LF_E5",
    "LowFlexElectrolysis+PtX": "LF_EPtX",
    "LowFlexCapacities": "LF_CAP",
    "LowFlexBEVDSM": "LF_BEV",
    "LowFlexBEVDSMOFF": "LF_BEV_OFF",
    
    # Battery variations
    "LowBattery75": "LB75",
    "LowBattery50": "LB50",
    "LowBattery25": "LB25",
    "LowBattery0": "LB0",
    
    # PtG / Electrolysis
    "LowPtG75": "LG75",
    "LowPtG50": "LG50",
    "LowPtG25": "LG25",
    
    # PtX
    "LowPtX75": "LX75",
    "LowPtX50": "LX50",
    "LowPtX25": "LX25",
    
    # PtH
    "LowPtH75": "LPH75",
    "LowPtH50": "LPH50",
    "LowPtH25": "LPH25",
    
    # H2 storage
    "LowH2Store75": "LH75",
    "LowH2Store50": "LH50",
    "LowH2Store25": "LH25",
    "LowH2Store0": "LH0",
    
    # Heat storage
    "LowHeatStore75": "HS75",
    "LowHeatStore50": "HS50",
    "LowHeatStore25": "HS25",
    "LowHeatStore0": "HS0",
    
    # Transmission
    "LowTransmission25": "LT25",
    "LowTransmission10": "LT10",
    
    # High flex variations
    "HighFlexIndustry": "HF_IND",
    "HighFlexIronAir": "HF_IA",
    "HighFlexBEV70": "HF_B70",
    "HighFlexBEV80": "HF_B80",
    "HighFlexBEV90": "HF_B90",

    # OtherWeatherYears variations
    "Base_2012": "BA2012",
    "Base_2013": "BA2013",
    "Base_2020": "BA2020",
    "Base_2023": "BA2023",

    # CAPEX sensitivities
    "Base_DoubleBattery" : "BA_2BAT",
    "Base_DoubleElectrolysis" : "BA_2ELEC",
    "HighFlexDoubleIronAir" : "HF_2IA",
    "HighFlexFiveIronAir" : "HF_5IA",

    # Ariadne scenarios
    "ExPol": "EP",
    "KN2045_Mix": "KN_MIX",
    "KN2045_Elek": "KN_EL",
    "KN2045_H2": "KN_H2",
    "KN2045_NFniedrig": "KN_NFL",
    "KN2045_NFhoch": "KN_NFH",
}


def aggregate_small_contributors(df, threshold=0.01):
    """
    Aggregate technologies contributing <1% across all columns into 'Other'
    Takes df with Technologies as index and granularity + year as columns
    """
    col_totals = df.abs().sum()
    mask = (df.abs() < threshold * col_totals).all(axis=1)
    result = df[~mask].copy()
    result.loc["Other"] = df[mask].sum()

    assert isclose(result.sum(), df.sum()).all(), "Sum mismatch after aggregation"

    return result

def df_to_png(df, filename="table.png"):

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.25*len(df)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   rowLabels=df.index,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


# definitions

resistive_heater = [
    "urban central resistive heater",
    "rural resistive heater",
    "urban decentral resistive heater",
]
gas_boiler = [
    "urban central gas boiler",
    "rural gas boiler",
    "urban decentral gas boiler",
]
heat_pump = [
    "urban central air heat pump",
    "rural air heat pump",
    "rural ground heat pump",
    "urban decentral air heat pump",
]
water_tanks_charger = [
    "urban central water tanks charger",
    "rural water tanks charger",
    "urban decentral water tanks charger",
]
water_tanks_discharger = [
    "urban central water tanks discharger",
    "rural water tanks discharger",
    "urban decentral water tanks discharger",
]
solar_thermal = [
    "urban decentral solar thermal",
    "urban central solar thermal",
    "rural solar thermal",
]

solar = ["solar", "solar-hsat"]
offwind = ["offwind-ac", "offwind-dc"]
h2_ocgt = ["H2 OCGT", "H2 retrofit OCGT"]

c1_groups = [resistive_heater, gas_boiler, heat_pump, solar, offwind, h2_ocgt]

c1_groups_name = [
    "resistive heater",
    "gas boiler",
    "heat pump",
    "solar",
    "offwind",
    "H2 OCGT",
]

scenario_colors = {
    # Main scenarios
    'LowFlex': 'navy',    
    'LowBattery': 'teal', 
    'Base': 'darkorange',        
    'HighFlex': 'purple',
    
    # LowFlex variations
    'LowFlex75_pminpu_FT_electrolysis0.9': '#1e3a8a',
    'LowFlex75_pminpu_not_electrolysis0.9': '#1e40af',
    'LowFlex75_pminpu_only_electrolysis0.9': '#1e50c7',
    'LowFlex75_pminpu_0.9': '#2563eb',
    'LowFlex75_pminpu_0.8': '#3b82f6',
    'LowFlex75_CapRestrictDEWide': '#60a5fa',
    'LowFlex75_excludeHeat_H2Store': '#93c5fd',
    'LowFlex75_excludeHeat': '#bfdbfe',
    'LowFlex75_exclude_pminpu': '#dbeafe',
    'LowFlex75_exclude_pminpu_onlyStores': '#eff6ff',
    'LowFlex75': '#1e293b',
    'LowFlex50': '#334155',
    'LowFlex50_pminpu0.9': '#475569',
    'LowFlex50_pminpu0.8': '#64748b',
    'LowFlex50HeatDecentral25': '#94a3b8',
    
    # Heat variations
    'LowFlexHeatModelMargin0.1': '#be123c',
    'LowFlexHeatModelMargin0.1pminpu0.9': '#e11d48',
    'LowFlexHeatModelMargin0.2': '#f43f5e',
    'LowFlexHeatModelMargin0.3': '#fb7185',
    'LowFlexHeatPMINPU50': '#fda4af',
    'LowFlexHeatPMINPU25': '#fecdd3',
    'LowFlexHeatMinStore': '#ffe4e6',
    'LowFlexBEVDSMOFF': '#881337',
    
    # Battery variations
    'LowBattery75': '#0d9488',
    'LowBattery50': '#14b8a6',
    'LowBattery25': '#2dd4bf',
    'LowBattery0': '#5eead4',
    
    # PtG / Electrolysis
    'LowPtG75': '#15803d',
    'LowPtG50': '#16a34a',
    'LowPtG25': '#22c55e',
    
    # PtX
    'LowPtX75': '#ca8a04',
    'LowPtX50': '#eab308',
    'LowPtX25': '#facc15',
    
    # PtH
    'LowPtH75': '#c2410c',
    'LowPtH50': '#ea580c',
    'LowPtH25': '#fb923c',
    
    # H2 storage
    'LowH2Store75': '#7c2d12',
    'LowH2Store50': '#9a3412',
    'LowH2Store25': '#c2410c',
    'LowH2Store0': '#ea580c',
    
    # Heat storage
    'LowHeatStore75': '#991b1b',
    'LowHeatStore50': '#dc2626',
    'LowHeatStore25': '#ef4444',
    'LowHeatStore0': '#f87171',
    
    # Transmission
    'LowTransmission25': '#4338ca',
    'LowTransmission10': '#6366f1',
    
    # High flex variations
    'HighFlexIndustry': '#6b21a8',
    'HighFlexIronAir': '#7c3aed',
    'HighFlexDLR': '#8b5cf6',
    'HighFlexBEV70': '#a78bfa',
    'HighFlexBEV80': '#c4b5fd',
    'HighFlexBEV90': '#ddd6fe',
    
    # Ariadne scenarios
    'KN2045_Mix': '#065f46',
    'KN2045_Elek': '#059669',
    'KN2045_H2': '#10b981',
    'KN2045_NFniedrig': '#34d399',
    'KN2045_NFhoch': '#6ee7b7',
}


# def collapse_small_columns(df, threshold=0.01, others_name="other"):
#     """
#     Collapse columns where all entries are below threshold share into 'other' column
#     Takes df with Technologies as columns and granularity + year as index
#     """
#     row_abs_sum = df.abs().sum(axis=1)
#     shares = df.abs().div(row_abs_sum, axis=0)

#     cols_keep = (shares >= threshold).any(axis=0)
#     cols_drop = ~cols_keep

#     df_out = df.loc[:, cols_keep].copy()
#     df_out[others_name] = df.loc[:, cols_drop].sum(axis=1)

#     assert isclose(df_out.sum(axis=1), df.sum(axis=1)).all(), "Sum mismatch after collapsing columns"

#     return df_out
