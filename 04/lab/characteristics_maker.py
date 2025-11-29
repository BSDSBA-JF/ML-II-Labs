import os
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_basic_character_attributes(
    url: str = "https://gi.yatta.moe/api/v2/en/avatar",
    file_path: str = "data/characters/character_attributes.csv",
) -> pd.DataFrame:
    """
    Fetches basic character attributes from the Yatta.moe API and returns a cleaned DataFrame.
    If a CSV file already exists at `file_path`, it reads from the file instead.

    Args:
        url (str): API endpoint to fetch character data.
        file_path (str): Path to save or read the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing cleaned character attributes
                      (name, weapon type, region, body type).
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    # Fetch JSON data
    data = requests.get(url).json()
    character_items = data["data"]["items"]
    df = pd.DataFrame.from_dict(character_items, orient="index")

    # Clean Weapon Type
    df["Weapon Type"] = df["weaponType"].replace(
        {
            "WEAPON_SWORD_ONE_HAND": "Sword",
            "WEAPON_CATALYST": "Catalyst",
            "WEAPON_CLAYMORE": "Claymore",
            "WEAPON_BOW": "Bow",
            "WEAPON_POLE": "Polearm",
        }
    )

    # Clean Ascension Bonus
    stat_mapping = {
        "FIGHT_PROP_CRITICAL_HURT": "Critical Damage",
        "FIGHT_PROP_CRITICAL": "Critical Rate",
        "FIGHT_PROP_HEAL_ADD": "Healing Bonus",
        "FIGHT_PROP_ELEMENT_MASTERY": "Elemental Mastery",
        "FIGHT_PROP_HP_PERCENT": "HP%",
        "FIGHT_PROP_ATTACK_PERCENT": "Attack%",
        "FIGHT_PROP_DEFENSE_PERCENT": "Defense%",
        "FIGHT_PROP_CHARGE_EFFICIENCY": "Energy Recharge",
        "FIGHT_PROP_PHYSICAL_ADD_HURT": "Physical Damage Bonus",
        "FIGHT_PROP_ELEC_ADD_HURT": "Electro Damage Bonus",
        "FIGHT_PROP_FIRE_ADD_HURT": "Pyro Damage Bonus",
        "FIGHT_PROP_WATER_ADD_HURT": "Hydro Damage Bonus",
        "FIGHT_PROP_ICE_ADD_HURT": "Cryo Damage Bonus",
        "FIGHT_PROP_WIND_ADD_HURT": "Anemo Damage Bonus",
        "FIGHT_PROP_ROCK_ADD_HURT": "Geo Damage Bonus",
        "FIGHT_PROP_GRASS_ADD_HURT": "Dendro Damage Bonus",
    }
    df["Ascension Bonus"] = df["specialProp"].replace(stat_mapping)

    # Capitalize Region, Body Type
    df["Region"] = df["region"].str.title()
    df["Body Type"] = df["bodyType"].str.title()

    relevant_cols = ["name", "Weapon Type", "Region", "Body Type"]

    # Exclude Travelers and Manekina/Manekin
    mask = ~df["name"].isin(["Traveler", "Manekina", "Manekin"])

    df_clean = df.loc[mask, relevant_cols]

    # Save CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_clean.to_csv(file_path, index=False)

    return df_clean


def get_character_ownership(
    version: int = 52, file_path: str = "data/characters/characters_ownership.csv"
) -> pd.DataFrame:
    """
    Retrieves character ownership and usage data from YSHelper API for a given version.
    Calculates estimated number of pulls per character. If the CSV file already exists,
    it reads from the file instead of fetching from the API.

    Args:
        version (int): Version number of the Abyss rank data.
        file_path (str): Path to save or read the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['name', 'star', 'use', 'own', 'pull_number'].
    """
    # If CSV exists, read and return
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    url = f"https://api.yshelper.com/ys/getAbyssRank.php?star=all&role=all&lang=en&version={version}"
    data = requests.get(url).json()

    chars = {}
    for rank_char in data["result"][0]:
        for char in rank_char["list"]:
            char_name = char["name"]
            chars[char_name] = char

    df_owns = pd.DataFrame(chars).T

    # Clean character names
    df_owns["name"] = df_owns["name"].replace(
        {"Ambor": "Amber", "Ayaka": "Kamisato Ayaka"}
    )

    # Calculate number of pulls
    c_rate_cols: List[str] = [
        "c0_rate",
        "c1_rate",
        "c2_rate",
        "c3_rate",
        "c4_rate",
        "c5_rate",
        "c6_rate",
    ]
    df_owns.loc[:, c_rate_cols] = df_owns[c_rate_cols] / 100
    df_owns["Pull Number"] = df_owns["own"] * sum(
        df_owns[c] * (i + 1) for i, c in enumerate(c_rate_cols)
    )
    df_owns["Pull Number"] = df_owns["Pull Number"].astype(int)

    # Select relevant columns
    df_final = df_owns[["name", "star", "use", "own", "Pull Number"]]
    df_final.columns = df_final.columns.str.title()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_final.to_csv(file_path, index=False)

    return df_final


def get_genshin_character_roles(
    file_path: str = "data/characters/character_roles.csv",
) -> pd.DataFrame:
    """
    Scrapes the Genshin Impact Fandom Wiki page for character role data,
    converts checkmarks and crosses to 1/0, and saves the cleaned table to CSV.

    Args:
        output_path (str): Path to save the cleaned CSV file.

    Returns:
        pd.DataFrame: Cleaned character role DataFrame with binary role indicators.
    """
    # If it exists, just read
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    url = "https://genshin-impact.fandom.com/wiki/Character_Role"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", class_="article-table")

    if len(tables) < 2:
        raise ValueError("Could not find more than one matching table.")

    # Use the second table
    target_table = tables[1]
    df = pd.read_html(StringIO(str(target_table)))[0]

    # Convert checkmarks/crosses to 1/0
    role_columns = df.columns[2:]  # skip first two columns (e.g., name, weapon)
    for col in role_columns:
        df[col] = df[col].apply(lambda x: {"✘": 0, "✔": 1}.get(x, 0))

    # create a mask
    is_not_traveler_manekim = (
        ~(df["Name"].str.contains("Traveler"))
        & (df["Name"] != "Manekina")
        & (df["Name"] != "Manekin")
    )
    df = df.loc[is_not_traveler_manekim]

    # Save to CSV
    df.to_csv(file_path, index=False)

    return df


def get_characteristics(
    filepath: str = "data/characters/characteristics.csv",
) -> pd.DataFrame:
    """
    Merges basic character attributes and character roles into a single DataFrame.
    If the CSV file already exists, it reads from the file instead of merging.

    Args:
        filepath (str): Path to save or read the merged characteristics CSV.

    Returns:
        pd.DataFrame: Merged character DataFrame with roles and attributes.
    """
    # If it exists, just read
    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    # If it doesnt, get the basic characteristics, roles, and wishes, line counts first
    df_basic_characteristics = get_basic_character_attributes()
    df_roles = get_genshin_character_roles()
    df_owns = get_character_ownership()
    df_line_count = pd.read_csv("data/line_count.csv")[["name", "line_count"]]

    # Merging time
    df_merged = df_roles.merge(
        df_basic_characteristics, left_on="Name", right_on="name"
    )

    df_merged = df_merged.merge(df_line_count, left_on="Name", right_on="name").drop(
        columns=["name_x", "name_y"]
    )

    df_merged = df_merged.merge(df_owns, on="Name")

    # Clean
    df_merged.rename(columns={"line_count": "Line Count"}, inplace=True)

    # Save merged DataFrame
    df_merged.to_csv(filepath, index=False)

    return df_merged