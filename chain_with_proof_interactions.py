import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from scipy.stats import chi2_contingency

# Define states
states = ["New Customer", "Engaged Customer", "Interested in Upgrade", "Requested Upgrade Info", "Upgraded"]

# Define transition probability matrix
transition_matrix = np.array([
    [0.6, 0.4, 0.0, 0.0, 0.0],  # New Customer
    [0.0, 0.7, 0.3, 0.0, 0.0],  # Engaged Customer
    [0.0, 0.0, 0.8, 0.2, 0.0],  # Interested in Upgrade
    [0.0, 0.0, 0.0, 0.9, 0.1],  # Requested Upgrade Info
    [0.0, 0.0, 0.0, 0.0, 1.0]   # Upgraded
])

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load the cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_state_index(state):
    return states.index(state)

def determine_state(sentence, states):
    state_scores = []
    for state in states:
        score = cross_encoder.predict([(sentence, state)])[0]
        state_scores.append(score)
    
    current_state = states[np.argmax(state_scores)]
    return current_state, np.max(state_scores)

def get_transition_probabilities(current_state, transition_matrix):
    current_index = get_state_index(current_state)
    probabilities = transition_matrix[current_index]
    return probabilities

# Define example dialogs
dialogs = {
    "Dialog 1": [
        "The customer is new and has just signed up.",
        "The customer seems quite engaged with our services.",
        "The customer has expressed interest in upgrading their plan.",
        "The customer has requested more information about the upgrade.",
        "The customer has successfully upgraded their plan."
    ],
    "Dialog 2": [
        "The customer is new and has just signed up.",
        "The customer seems quite engaged with our services.",
        "The customer has requested more information about the upgrade.",
        "The customer has expressed interest in upgrading their plan.",
        "The customer has successfully upgraded their plan."
    ],
    "Dialog 3": [
        "The customer seems quite engaged with our services.",
        "The customer has requested more information about the upgrade.",
        "The customer is new and has just signed up.",
        "The customer has successfully upgraded their plan.",
        "The customer has expressed interest in upgrading their plan."
    ],
    "Dialog 4": [
        "The customer has expressed interest in upgrading their plan.",
        "The customer has successfully upgraded their plan.",
        "The customer is new and has just signed up.",
        "The customer seems quite engaged with our services.",
        "The customer has requested more information about the upgrade."
    ],
    "Dialog 5": [
        "The customer has requested more information about the upgrade.",
        "The customer has requested more information about the upgrade.",
        "The customer has requested more information about the upgrade.",
        "The customer has requested more information about the upgrade.",
        "The customer has requested more information about the upgrade."
    ],
    "Dialog 6": [
        "The customer has requested more information about the upgrade.",
        "The customer has expressed interest in upgrading their plan.",
        "The customer seems quite engaged with our services.",
        "The customer is new and has just signed up.",
        "The customer has successfully upgraded their plan."
    ]
}

'''
Processing Each Dialog:

Each dialog is processed sentence by sentence.
The state is determined for each sentence, and transitions between consecutive sentences are recorded.
State counts and transition counts are updated for the entire dialog.
Transition Probabilities Calculation:

The transition counts are normalized to obtain observed transition probabilities for the entire dialog.
The observed transition probabilities are compared against the expected transition probabilities (defined by your transition matrix) using a chi-squared test.
Chi-Squared Test:

The chi-squared test compares the flattened observed transition probabilities and expected transition probabilities.
It calculates the chi-squared statistic and p-value to determine if there is a significant difference between the observed and expected probabilities.
'''

for dialog_name, dialog in dialogs.items():
    print(f"\n{dialog_name}:")

    # Initialize counters for each state
    state_counts = {state: 0 for state in states}

    # Initialize transition counts
    transition_counts = np.zeros((len(states), len(states)))

    # Process the dialog
    previous_state = None
    for sentence in dialog:
        current_state, confidence = determine_state(sentence, states)
        state_counts[current_state] += 1
        print(f"Sentence: '{sentence}' => Current State: {current_state} with confidence: {confidence}")

        if previous_state is not None:
            index_from = get_state_index(previous_state)
            index_to = get_state_index(current_state)
            transition_counts[index_from, index_to] += 1
        
        previous_state = current_state

    print("\nState Counts:")
    for state, count in state_counts.items():
        print(f"  {state}: {count}")

    # Normalize to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    observed_probabilities = transition_counts / row_sums

    print("\nObserved Transition Probabilities:")
    for i, state_from in enumerate(states):
        for j, state_to in enumerate(states):
            print(f"  From '{state_from}' to '{state_to}': {observed_probabilities[i, j]}")

    # Add a small value to avoid zero frequencies
    epsilon = 1e-6
    adjusted_transition_matrix = transition_matrix + epsilon
    row_sums_adjusted = adjusted_transition_matrix.sum(axis=1, keepdims=True)
    expected_probabilities = adjusted_transition_matrix / row_sums_adjusted
    expected_flat_adjusted = expected_probabilities.flatten()

    # Flatten observed probabilities
    observed_flat = observed_probabilities.flatten()

    # Use chi-squared test to check goodness-of-fit
    chi2_stat, p_value, _, _ = chi2_contingency([observed_flat, expected_flat_adjusted])

    print(f"\nChi-Squared Statistic: {chi2_stat}")
    print(f"P-Value: {p_value}")

    # Check if the p-value indicates a good fit
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("The observed transition probabilities significantly differ from the expected probabilities.")
    else:
        print("The observed transition probabilities do not significantly differ from the expected probabilities.")
