import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()   

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        sequenceLength = len(seq)
        probability = np.zeros((self.N, sequenceLength))

        for stateIndex in range(self.N):
            probability[stateIndex, 0] = self.pi[stateIndex] * self.B[stateIndex, self.emissions_dict[seq[0]]]

        for emission in range(1, sequenceLength):
            for state in range(self.N):
                maxProb = 0
                for prevState in range(self.N):
                    maxProb = max(
                        probability[prevState, emission - 1] * self.A[prevState, state] * self.B[state, self.emissions_dict[seq[emission]]],
                        maxProb
                    )
                probability[state, emission] = maxProb
        
        reverseDict = { value: key for key, value in self.states_dict.items() }
        return [reverseDict[i] for i in np.argmax(probability, axis = 0)]