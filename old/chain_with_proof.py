import numpy as np
from scipy import stats
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Define a tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define your states
states = ["New Customer", "Returning Customer", "Loyal Customer"]

# Simulate some example sentences
example_sentences = [
    "The customer is new and has just signed up.",
    "The customer has made multiple purchases and is returning.",
    "The customer frequently makes purchases and is considered loyal."
]

# Simulate transitions based on example sentences
def simulate_transitions(sentences, num_simulations=1000):
    transitions = {state: {s: 0 for s in states} for state in states}
    
    for _ in range(num_simulations):
        sentence = np.random.choice(sentences)
        current_state = determine_state(sentence)
        next_state = np.random.choice(states)
        if current_state in transitions:
            transitions[current_state][next_state] += 1
    
    return transitions

def determine_state(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        state_index = torch.argmax(probs).item()
    
    return states[state_index]

# Define transition matrix (example probabilities)
transition_matrix = np.array([
    [0.2, 0.5, 0.3],  # From New Customer
    [0.3, 0.4, 0.3],  # From Returning Customer
    [0.1, 0.3, 0.6]   # From Loyal Customer
])

def get_state_index(state):
    return states.index(state)

def verify_probabilities(transitions, transition_matrix, num_simulations):
    for state, counts in transitions.items():
        observed = np.array([counts[s] for s in states])
        total_observed = np.sum(observed)
        if total_observed == 0:
            print(f"No observations for state '{state}'")
            continue

        expected = np.array([transition_matrix[get_state_index(state)][get_state_index(s)] * num_simulations for s in states])
        
        # Avoid divisions by zero or negative values in expected counts
        expected = np.maximum(expected, 1e-10)
        
        try:
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            print(f"Chi-square test for state '{state}': chi2 = {chi2}, p-value = {p_value}")

            if p_value < 0.05:
                print(f"Warning: Probabilities for state '{state}' do not match expected values (p-value = {p_value}).")
        except ValueError as e:
            print(f"ValueError during chi-square test for state '{state}': {e}")

# Example simulation and verification
num_simulations = 1000
transitions = simulate_transitions(example_sentences, num_simulations)
verify_probabilities(transitions, transition_matrix, num_simulations)
