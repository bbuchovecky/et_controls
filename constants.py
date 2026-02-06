"""Constants for the FHIST PPE."""

PARAMETERS = [
    "dleaf",
    "d_max",
    "maximum_leaf_wetted_fraction",
    "fff",
    "medlynslope",
    "medlynintercept",
    "Jmaxb0" "kmax",
    "psi50",
    "FUN_fracfixers",
    "leafcn",
    "lmrha",
    "KCN",
    "ACCLIM_SF",
]

MEMBER_DICT = {
    "ACCLIM_SF": {"max": "28", "min": "27"},
    "FUN_fracfixers": {"max": "20", "min": "19"},
    "Jmaxb0": {"max": "14", "min": "13"},
    "KCN": {"max": "26", "min": "25"},
    "d_max": {"max": "4", "min": "3"},
    "default": {"default": "0"},
    "dleaf": {"max": "2", "min": "1"},
    "fff": {"max": "8", "min": "7"},
    "kmax": {"max": "16", "min": "15"},
    "leafcn": {"max": "22", "min": "21"},
    "lmrha": {"max": "24", "min": "23"},
    "maximum_leaf_wetted_fraction": {"max": "6", "min": "5"},
    "medlynintercept": {"max": "12", "min": "11"},
    "medlynslope": {"max": "10", "min": "9"},
    "psi50": {"max": "18", "min": "17"},
}

# https://medialab.github.io/iwanthue/
COLORMAP = {
    "ACCLIM_SF": "#b14e83",
    "FUN_fracfixers": "#838bb5",
    "Jmaxb0": "#73915b",
    "KCN": "#8543a5",
    "d_max": "#c24552",
    "default": "#000000",
    "dleaf": "#706c6c",
    "fff": "#94644c",
    "kmax": "#62ba5b",
    "leafcn": "#637dd2",
    "lmrha": "#9843e0",
    "maximum_leaf_wetted_fraction": "#d25c28",
    "medlynintercept": "#a4cc32",
    "medlynslope": "#b29543",
    "psi50": "#5bb0a1",
}

LS = {
    "default": "-",
    "min": "--",
    "max": "-",
}
