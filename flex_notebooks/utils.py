# Define flexible technologies
flex_techs = {
    "no": "",
    "interconnectors": ["AC", "DC"],
    "coal": ["lignite", "coal"],
    "gas": ["OCGT", "CCGT"],
    "biogas": ["biogas"],
    "oil": ["oil"],
    "nuclear": ["nuclear"],
    "solid biomass": ["solid biomass"],
    "PHS": ["PHS"],
    "H2 OCGT": ["H2 OCGT", "H2 retrofit OCGT"],
    "battery": ["battery discharger"],
    "coal CHP": ["urban central coal CHP", "urban central lignite CHP"],
    "gas CHP": ["urban central gas CHP", "urban central gas CHP CC"],
    "H2 CHP": ["urban central H2 CHP", "urban central H2 retrofit CHP"],
    "biomass CHP": [
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
    ],
    "waste CHP": ["waste CHP", "waste CHP CC"],
    "hydro": ["hydro"],
    "Fuel Cell": ["H2 Fuel Cell"],
}

# Define flexible technologies
flex_techs_supply = {
    "no": "",
    "interconnectors supply": ["AC", "DC"],
    "import": ["import"],
    "coal": ["lignite", "coal"],
    "gas": ["OCGT", "CCGT"],
    "biogas": ["biogas"],
    "oil": ["oil"],
    "nuclear": ["nuclear"],
    "solid biomass": ["solid biomass"],
    "PHS discharging": ["PHS"],
    "H2 OCGT": ["H2 OCGT", "H2 retrofit OCGT"],
    "battery discharger": ["battery discharger", "home battery discharger"],
    "coal CHP": ["urban central coal CHP", "urban central lignite CHP"],
    "gas CHP": ["urban central gas CHP", "urban central gas CHP CC"],
    "H2 CHP": ["urban central H2 CHP", "urban central H2 retrofit CHP"],
    "biomass CHP": [
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
    ],
    "waste CHP": ["waste CHP", "waste CHP CC"],
    "hydro": ["hydro"],
    "Fuel Cell": ["H2 Fuel Cell"],
}

flex_techs_demand = {
    "interconnectors demand": ["AC", "DC"],
    "export": ["export"],
    "H2 Electrolysis": ["H2 Electrolysis"],
    "BEV charger": ["BEV charger"],
    "PHS charging": ["PHS"],
    "battery charger": ["battery charger", "home battery charger"],
    "heat pump": [
        "rural ground heat pump",
        "rural air heat pump",
        "urban decentral air heat pump",
        "urban central air heat pump",
    ],
    "resistive heater": [
        "rural resistive heater",
        "urban decentral resistive heater",
        "urban central resistive heater",
    ],
    "methanolisation": ["methanolisation"],
}

carriers_in_german = {
    "DC": "Gleichstrom",
    "AC": "Wechselstrom",
    "hydro": "Wasserkraft (Reservoir & Damm)",
    "offwind-ac": "Offshore-Wind (AC)",
    "offwind-dc": "Offshore-Wind (DC)",
    "solar": "Solar",
    "solar-hsat": "Solar (HSAT)",
    "onwind": "Onshore-Wind",
    "PHS": "Pumpspeicherkraftwerk",
    "ror": "Laufwasserkraft",
    "": "",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "uranium": "Uran",
    "none": "keine",
    "co2": "CO2",
    "co2 stored": "CO2 gespeichert",
    "co2 sequestered": "CO2 sequestriert",
    "gas": "Gas",
    "H2": "Wasserstoffspeicher",
    "battery": "Batteriespeicher",
    "EV battery": "Elektrofahrzeug-Batterie",
    "urban central heat": "Zentrale städtische Heizung",
    "urban central water tanks": "Zentrale städtische Wassertanks",
    "urban central solar thermal": "Zentrale städtische Solarthermie",
    "biogas": "Biogas",
    "solid biomass": "Biomasse",
    "methanol": "Methanol",
    "home battery": "Hausbatterie",
    "rural heat": "Ländliche Wärme",
    "rural solar thermal": "Ländliche Solarthermie",
    "rural water tanks": "Ländliche Wassertanks",
    "urban decentral heat": "Dezentrale städtische Heizung",
    "urban decentral solar thermal": "Dezentrale städtische Solarthermie",
    "urban decentral water tanks": "Dezentrale städtische Wassertanks",
    "oil primary": "Primäröl",
    "solar rooftop": "Solar-Dach",
    "urban central heat vent": "Zentrale städtische Wärmeentlüftung",
    "gas primary": "Primärgas",
    "solid biomass for industry": "Biomasse (Industrie)",
    "shipping oil": "Schiffsöl",
    "agriculture machinery oil": "Landwirtschaftsmaschinenöl",
    "naphtha for industry": "Naphtha für die Industrie",
    "land transport oil": " Öl (Transport)",
    "kerosene for aviation": "Kerosin für die Luftfahrt",
    "non-sequestered HVC": "Nicht-sequestriertes HVC",
    "shipping methanol": "Methanol für Schifffahrt",
    "gas for industry": "Gas (Industrie)",
    "low voltage": "Niederspannung",
    "industry methanol": "Methanol (Industrie)",
    "coal for industry": "Kohle (Industrie)",
    "process emissions": "Prozessemissionen",
    "industry electricity": "Industrieelektrizität",
    "low-temperature heat for industry": "Niedertemperaturwärme (Industrie)",
    "agriculture electricity": "Landwirtschaftliche Elektrizität",
    "electricity": "Elektrizität",
    "land transport EV": "Elektrofahrzeuge (Transport)",
    "agriculture heat": "Landwirtschaftliche Wärme",
    "urban central air heat pump": "Zentrale städtische Luftwärmepumpe",
    "electricity distribution grid": "Stromverteilungsnetz",
    "battery charger": "Batterie Laden",
    "waste CHP": "Müll-KWK",
    "urban central water tanks discharger": "Entladung zentraler städtischer Wassertanks",
    "rural water tanks discharger": "Entladung ländlicher Wassertanks",
    "urban decentral biomass boiler": "Dezentrale städtische Biomassekessel",
    "BEV charger": "E-Fahrzeug Laden",
    "rural ground heat pump": "Erdwärmepumpe",
    "Fischer-Tropsch": "Fischer-Tropsch",
    "urban decentral water tanks discharger": "Entladung dezentraler städtischer Wassertanks",
    "urban central gas CHP": "Gas-KWK",
    "Sabatier": "Sabatier-Prozess",
    "gas compressing": "Gasverdichtung",
    "home battery charger": "Hausbatterie Laden",
    "battery discharger": "Batterie Entladung",
    "H2 pipeline retrofitted": "Nachgerüstete Wasserstoffpipeline",
    "rural resistive heater": "Ländlicher Widerstandsheizer",
    "urban central water tanks charger": "Ladung zentraler städtischer Wassertanks",
    "urban central solid biomass CHP CC": "Biomasse-KWK mit CO2-Abscheidung",
    "rural air heat pump": "Ländliche Luftwärmepumpe",
    "DAC": "Direkte CO2-Abscheidung",
    "urban decentral air heat pump": "Dezentrale städtische Luftwärmepumpe",
    "waste CHP CC": "Müll-KWK mit CO2-Abscheidung",
    "biomass to liquid CC": "Biomasse zu Flüssigkeit mit CO2-Abscheidung",
    "HVC to air": "HVC in die Luft",
    "methanolisation": "Methanolisation",
    "gas for industry CC": "Gas mit CO2-Abscheidung (Industrie)",
    "H2 pipeline": "Wasserstoffpipeline",
    "solid biomass for industry CC": "Biomasse mit CO2-Abscheidung (Industrie)",
    "urban decentral gas boiler": "Dezentrale städtische Gaskessel",
    "urban central gas CHP CC": "Gas-KWK mit CO2-Abscheidung",
    "rural gas boiler": "Ländlicher Gaskessel",
    "process emissions CC": "Prozessemissionen mit CO2-Abscheidung",
    "urban decentral water tanks charger": "Ladung dezentraler städtischer Wassertanks",
    "biogas to gas CC": "Biogas zu Gas mit CO2-Abscheidung",
    "urban decentral resistive heater": "Dezentrale städtische Widerstandsheizer",
    "biogas to gas": "Biogas zu Gas",
    "OCGT": "Gas (OCGT)",
    "urban central solid biomass CHP": "Biomasse-KWK",
    "urban central gas boiler": "Zentraler städtischer Gaskessel",
    "urban central resistive heater": "Zentraler städtischer Widerstandsheizer",
    "oil refining": "Ölraffinierung",
    "rural biomass boiler": "Ländlicher Biomassekessel",
    "SMR": "Dampfreformierung",
    "biomass to liquid": "Biomasse zu Flüssigkeit",
    "rural water tanks charger": "Ladung ländlicher Wassertanks",
    "home battery discharger": "Hausbatterie Entladung",
    "H2 Store": "Wasserstoffspeicher",
    "renewable oil": "Erneuerbares Öl",
    "renewable gas": "Erneuerbares Gas",
    "CCGT": "Gas (CCGT)",
    "nuclear": "Kernenergie",
    "gas CHP": "Gas-KWK",
    "gas CHP CC": "Gas KWK mit CO2-Abscheidung",
    "urban central coal CHP": "Steinkohle-KWK",
    "coal CHP": "Steinkohle-KWK",
    "Electricity trade": "Stromhandel",
    "urban central gas CHP": "Gas-KWK",
    "urban central biomass CHP": " Biomasse-KWK",
    "biomass CHP": "Biomasse-KWK",
    "air heat pump": "Luftwärmepumpe",
    "Electricity load": "Elektrizitätslast",
    "methanolisation": "Methanolisierung",
    "resistive heater": "Widerstandsheizer",
    "gas boiler": "Gaskessel",
    "H2 Electrolysis": "Elektrolyse",
    "H2 Fuel Cell": "Brennstoffzelle (Strom)",
    "H2 for industry": "H2 für Industrie",
    "H2 OCGT": "Wasserstoff (OCGT)",
    "H2 CHP": "H2 KWK",
    "land transport fuel cell": "Brennstoffzelle (Verkehr)",
    "other": "Sonstige",
    "SMR CC": "Dampfreformierung mit CCS",
    "H2 pipeline (new)": "H2 Pipeline (Neubau)",
    "H2 pipeline (repurposed)": "H2 Pipeline (Umstellung)",
    "H2 pipeline (Kernnetz)": "H2 Pipeline (Kernnetz)",
    "heat pump": "Wärmepumpe",
    "urban central H2 CHP": "Wasserstoff-KWK",
    "urban central H2 retrofit CHP": "Wasserstoff-KWK (Umrüstung)",
    "urban central oil CHP": "Öl-KWK",
    "urban central lignite CHP": "Braunkohle-KWK",
    "H2 retrofit OCGT": "Wasserstoff (OCGT;Umrüstung)",
}


# definitions

resistive_heater = ['urban central resistive heater', 'rural resistive heater','urban decentral resistive heater']
gas_boiler = ['urban central gas boiler', 'rural gas boiler','urban decentral gas boiler']
heat_pump = ['urban central air heat pump', 'rural air heat pump','rural ground heat pump', 'urban decentral air heat pump']
water_tanks_charger = ['urban central water tanks charger', 'rural water tanks charger', 'urban decentral water tanks charger']
water_tanks_discharger = ['urban central water tanks discharger','rural water tanks discharger', 'urban decentral water tanks discharger']
solar_thermal = [ "urban decentral solar thermal", "urban central solar thermal", "rural solar thermal"]

carrier_renaming = {
    'urban central solid biomass CHP CC': 'biomass CHP CC',
    'urban central solid biomass CHP': 'biomass CHP',
    'urban central gas CHP': 'gas CHP',
    'urban central gas CHP CC': 'gas CHP CC',
    'urban central coal CHP': 'coal CHP',
    'urban central lignite CHP': 'lignite CHP',
    'urban central air heat pump': 'air heat pump',
    'urban central resistive heater': 'resistive heater'
}

carrier_renaming_reverse = {
    'biomass CHP CC': 'urban central solid biomass CHP CC',
    'biomass CHP' :'urban central solid biomass CHP' ,
    'gas CHP': 'urban central gas CHP' ,
    'gas CHP CC' : 'urban central gas CHP CC',
    'coal CHP':   'urban central coal CHP',
    'lignite CHP':  'urban central lignite CHP',
    'air heat pump' : 'urban central air heat pump',
    'resistive heater': 'urban central resistive heater'
}

def get_condense_sum(df, groups, groups_name, return_original=False):
    """
    return condensed df, that has been groupeb by condense groups
    Arguments:
        df: df you want to condense (carriers have to be in the columns)
        groups: group lables you want to condense on
        groups_name: name of the new grouped column
        return_original: boolean to specify if the original df should also be returned
    Returns:
        condensed df
    """
    result = df

    for group, name in zip(groups, groups_name):
        # check if carrier are in columns
        bool = [c in df.columns for c in group]
        # updated to carriers within group that are in columns
        group = list(compress(group, bool))

        result[name] = df[group].sum(axis=1)
        group_to_drop = [g for g in group if g != name]
        result.drop(group_to_drop, axis=1, inplace=True)

    if return_original:
        return result, df

    return result