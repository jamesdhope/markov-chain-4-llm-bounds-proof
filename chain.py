import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

# Define states
states = ["New Customer", "Engaged Customer", "Interested in Upgrade", "Requested Upgrade Info", "Upgraded"]

# Define transition probability matrix
# Each row corresponds to a state and each column corresponds to the probability of transitioning to the next state
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

# Define an example sentence
example_sentence = "The customer has shown interest in upgrading their subscription."

# Determine the current state based on the example sentence
current_state, confidence = determine_state(example_sentence, states)

print(f"Current State: {current_state} with confidence: {confidence}")

# Determine the probabilities of transitioning to all other states
transition_probabilities = get_transition_probabilities(current_state, transition_matrix)
print(f"Transition probabilities from '{current_state}':")
for state, probability in zip(states, transition_probabilities):
    print(f"  To '{state}': {probability}")
