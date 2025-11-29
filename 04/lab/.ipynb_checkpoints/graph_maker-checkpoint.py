import requests          # for fetching JSON from API
from urllib.parse import quote  # for URL encoding character names
import json              # to parse JSON strings
from itertools import combinations  # to compute pairwise combinations
from collections import defaultdict  # for default float dicts
from tqdm import tqdm    # for progress bars
import networkx as nx    # to build graphs

def create_mapping():
    """Creates a mapping from avatar image URLs to English character names."""
    image_english_mapping = {}
    url = "https://api.yshelper.com/ys/getAbyssRank.php"
    data = requests.get(url).json()

    for char_info in data["has_list"]:
        name = char_info["name"]
        image_link = char_info["avatar"]

        # Match Chinese name with English equivalent
        cn_to_ens = data["select_list"]
        for cn_to_en in cn_to_ens:
            cn = cn_to_en["title"][:-2]
            if cn == name:
                en = cn_to_en["value"]
                image_english_mapping[image_link] = en
                break  # Stop once matched

    # Add default mapping for Traveler
    image_english_mapping.update(
        {
            (
                "https://upload-bbs.mihoyo.com/game_record/genshin/character_icon/"
                "UI_AvatarIcon_PlayerGirl.png"
            ): "Traveler"
        }
    )

    # Add nefer too
    image_english_mapping.update(
        {
            (
                "https://inews.gtimg.com/om_bt/OZBabi2uXPQ10bwzQ6OP9abQ0YQvcyaSmMYn4UWERmFooAA/0"
            ): "Nefer"
        }
    )

    return image_english_mapping

image_english_mapping = create_mapping()


def get_character_names():
    """Get the character names"""
    url = "https://api.yshelper.com/ys/getAbyssRank.php"
    data = requests.get(url).json()

    char_names = []
    for char_json in data["select_list"][1:]:
        char_names.append(char_json["value"])

    return char_names

char_names = get_character_names()

def get_team_usage_per_char(char_name, version):
    """
    Given the character name and version, return the top 100 or fewer
    teams that include the character (from result[3]).
    """
    base_url = "https://api.yshelper.com/ys/getAbyssRank.php"
    url = f"{base_url}?star=all&role={quote(char_name)}" f"&lang=en&version={version}"

    team_data = requests.get(url).json()["result"][3]

def get_team_per_version(version, char_names):
    """Given the version and a list of character names, 
    get all the possible teams in a list with each elemenet as a json str"""
    possible_teams = set()

    # loop over all the characters
    for char in tqdm(char_names, desc="Fetching team usage"):
        # get the teams
        char_team_usage = get_team_usage_per_char(char, version)

        # if there are no teams, then skip this character
        if not char_team_usage:
            continue

        # add to a set all the possible teams
        for team in char_team_usage:
            possible_teams.add(json.dumps(team, sort_keys=True))
    return list(possible_teams)

def get_use_has_cooccur(teams, image_english_mapping):
    """
    Compute two pairwise co-occurrence dictionaries:
        1. co_occur_use: weighted by team["use"]
        2. co_occur_has: weighted by team["has"]

    Purpose:
        Later compute pairwise ratios: co_occur_use / co_occur_has
        → "Among players who COULD run this pair, how often do they DO run the pair?"

    Parameters
    ----------
    teams : iterable of JSON strings
    image_english_mapping : dict

    Returns
    -------
    co_occur_use : dict[(char1,char2) → float]
    co_occur_has : dict[(char1,char2) → float]
    """

    co_occur_use = defaultdict(float)
    co_occur_has = defaultdict(float)

    for team_json in tqdm(teams, desc="Building co-occurrence (use & has)"):
        team = json.loads(team_json)

        # Extract names
        char_names = []
        for c in team["role"]:
            avatar = c["avatar"]
            name = image_english_mapping.get(avatar)
            if name:
                char_names.append(name)
        
        if len(char_names) < 2:
            continue

        # Team-level weights
        use_w = team.get("use", 0)
        has_w = team.get("has", 0)

        # Add to pairwise co-occurrence
        for a, b in combinations(sorted(char_names), 2):
            co_occur_use[(a, b)] += use_w
            co_occur_has[(a, b)] += has_w

    return co_occur_use, co_occur_has

def get_graph(co_occur_use, co_occur_has):
    """
    Create a NetworkX graph where each edge contains two weights:
    - 'weight_use' : co-occurrence based on team usage
    - 'weight_has' : co-occurrence based on player ownership
    
    Parameters
    ----------
    co_occur_use : dict
        Absolute number of Usage-based co-occurrence values.

    co_occur_has : dict
        Absolute number of Ownership-based co-occurrence values.

    Returns
    -------
    networkx.Graph
        Undirected, with edges containing:
            - weight_use
            - weight_has
    """

    # union of keys
    all_pairs = set(co_occur_use.keys()) | set(co_occur_has.keys())

    G = nx.Graph()

    for char1, char2 in all_pairs:
        use_val = co_occur_use.get((char1, char2), 0)
        has_val = co_occur_has.get((char1, char2), 0)

        # Skip edges where both weights are 0
        if not (use_val or has_val):
            continue

        G.add_edge(
            char1, char2,
            weight_use=use_val,
            weight_has=has_val
        )

    return G

def create_graph(version, char_names=char_names, image_english_mapping=image_english_mapping):
    """
    Generate a co-occurrence graph for a specific game version with both
    usage-based and ownership-based edge weights.

    This function orchestrates the full pipeline:
    1. Fetch team compositions for the given version.
    2. Compute pairwise co-occurrence weights for all character pairs,
       both by 'use' and by 'has'.
    3. Build a NetworkX graph where each edge contains:
       - weight_use : co-occurrence based on team usage
       - weight_has : co-occurrence based on ownership

    Parameters
    ----------
    version : int or str
        The game version to analyze. Used by the webscraper to retrieve team data.
    char_names : list of str, optional
        List of all character names to scrape. Defaults to the global `char_names`.
    image_english_mapping : dict, optional
        Maps avatar identifiers to English character names for standardization.

    Returns
    -------
    networkx.Graph
        An undirected graph where:
        - Nodes are characters.
        - Edges represent pairwise co-occurrence.
        - Each edge has attributes:
            - weight_use : sum of team 'use' counts for the pair
            - weight_has : sum of team 'has' counts for the pair
    """

    # Fetch all possible teams for the version
    possible_teams = get_team_per_version(version, char_names)

    # Compute co-occurrence dictionaries (use & has)
    co_occur_use, co_occur_has = get_use_has_cooccur(possible_teams, image_english_mapping)

    # Build the graph with both weights
    return get_graph(co_occur_use, co_occur_has)
