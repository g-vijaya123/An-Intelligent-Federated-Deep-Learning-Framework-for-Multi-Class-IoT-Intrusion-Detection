"""
Human-readable information for UNSW-NB15 attack categories.

This module converts the predicted attack category into a simple
description that can be displayed to the user.
"""

ATTACK_INFO = {

    "Normal": {
        "type": "Normal Traffic",
        "description": (
            "The network traffic appears to be legitimate "
            "and no attack is detected."
        )
    },

    "Fuzzers": {
        "type": "Fuzzing Attack",
        "description": (
            "The attacker sends unexpected or random input "
            "to find weaknesses in a system or application."
        )
    },

    "Analysis": {
        "type": "Analysis Attack",
        "description": (
            "The attacker performs activities such as scanning "
            "or probing to collect information about the target system."
        )
    },

    "Backdoor": {
        "type": "Backdoor Attack",
        "description": (
            "The attacker attempts to gain unauthorized access "
            "through a hidden or previously created access mechanism."
        )
    },

    "DoS": {
        "type": "Denial of Service",
        "description": (
            "The attacker attempts to make a system or service unavailable "
            "by overwhelming or disrupting it."
        )
    },

    "Exploits": {
        "type": "Exploit Attack",
        "description": (
            "The attacker attempts to use a vulnerability in software "
            "or a service to perform unauthorized actions."
        )
    },

    "Generic": {
        "type": "Generic Attack",
        "description": (
            "The traffic shows a general attack pattern that may affect "
            "network or cryptographic security mechanisms."
        )
    },

    "Reconnaissance": {
        "type": "Reconnaissance Attack",
        "description": (
            "The attacker gathers information about hosts, services, ports, "
            "or the network before attempting further attacks."
        )
    },

    "Shellcode": {
        "type": "Shellcode Attack",
        "description": (
            "The attacker attempts to execute malicious low-level code "
            "on the target system."
        )
    },

    "Worms": {
        "type": "Worm Attack",
        "description": (
            "The attack involves malicious software that can spread "
            "automatically between vulnerable systems."
        )
    },
}


def get_attack_info(attack_name):
    """
    Return information about a predicted attack.

    Parameters:
        attack_name: Predicted UNSW-NB15 class name.

    Returns:
        Dictionary containing attack type and description.
    """

    # Remove possible whitespace from dataset labels
    attack_name = str(attack_name).strip()

    if attack_name in ATTACK_INFO:
        return ATTACK_INFO[attack_name]

    return {
        "type": attack_name,
        "description": (
            "No additional description is currently available "
            "for this predicted category."
        )
    }


def display_attack_info(attack_name):
    """Display predicted attack information."""

    info = get_attack_info(attack_name)

    print("=" * 60)
    print("ATTACK EXPLANATION")
    print("=" * 60)

    print(f"Predicted Category : {attack_name}")
    print(f"Attack Type        : {info['type']}")
    print(f"Description        : {info['description']}")

    print("=" * 60)


# ==================================================
# Test
# ==================================================

if __name__ == "__main__":

    test_attack = "DoS"

    display_attack_info(test_attack)