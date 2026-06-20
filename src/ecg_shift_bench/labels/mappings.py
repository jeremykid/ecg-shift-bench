"""Dataset-native to canonical label mappings.

RBBB and LBBB intentionally merge complete and incomplete bundle branch block
where a dataset provides those distinctions. SPH ``1dAVB`` is an approximation
based on its prolonged-PR-interval code. ``ST`` always means sinus tachycardia,
never an ST-segment abnormality.
"""

LABEL_MAP: dict[str, dict[str, list[str]]] = {
    "PTBXL": {
        "AF": ["AFIB"],
        "RBBB": ["CRBBB", "IRBBB"],
        "LBBB": ["CLBBB", "ILBBB"],
        "1dAVB": ["1AVB"],
        "SB": ["SBRAD"],
        "ST": ["STACH"],
    },
    "CHAPMAN": {
        "AF": ["AFIB"],
        "RBBB": ["RBBB"],
        "LBBB": ["LBBB"],
        "1dAVB": ["1AVB"],
        "SB": ["SB"],
        "ST": ["ST"],
    },
    "SPH": {
        "AF": ["50", "50+346", "50+347"],
        "RBBB": ["106"],
        "LBBB": ["104"],
        "1dAVB": ["82"],
        "SB": ["22"],
        "ST": ["21"],
    },
    "CODE15": {
        "AF": ["AF"],
        "RBBB": ["RBBB"],
        "LBBB": ["LBBB"],
        "1dAVB": ["1dAVb"],
        "SB": ["SB"],
        "ST": ["ST"],
    },
}
