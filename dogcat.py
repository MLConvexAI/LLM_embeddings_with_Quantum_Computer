import numpy as np
import json
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile  


def calculate_complex_similarity_exact(vectora, vectorb):
    """
    Calculates the complex cosine similarity between two vectors.
    """

    if len(vectora) != len(vectorb):
        raise ValueError("Vectors must have the same dimension")

    # Calculate the dot product of the vectors and their complex conjugates
    numerator = np.sum(np.conjugate(vectora) * vectorb)  

    # Calculate the magnitudes of the vectors
    magnitude_a = np.linalg.norm(vectora)
    magnitude_b = np.linalg.norm(vectorb)

    # Calculate the cosine similarity
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  
    else:
        similarity = numerator / (magnitude_a * magnitude_b)
        return similarity


def create_converted_vectors(vectora, vectorb):
    """
    Creates new vectors vectora_q and vectorb_q based on the conversion formulas.
    """

    if len(vectora) != len(vectorb):
        raise ValueError("Vectors must have the same length.")

    vectora_q = []
    vectorb_q = []

    for i in range(len(vectora)):
        a = vectora[i]
        b = vectorb[i]
        denominator = np.sqrt(abs(a)**2 + abs(b)**2)

        if denominator == 0:
            aq_i = 0  
            bq_i = 0  
        else:
            aq_i = a / denominator
            bq_i = b / denominator

        vectora_q.append(aq_i)
        vectorb_q.append(bq_i)

    return vectora_q, vectorb_q

def calculate_qubit_states_theoretical(vectora, vectorb):
    """
    Calculates qubit states based on the provided formulas and prints the results.
    Creates 2 qubits for each element in the input vectors.
    """

    if len(vectora) != len(vectorb):
        raise ValueError("Vectors must have the same length.")

    results = {}
    n = len(vectora)
    for i in range(n):
        a = vectora[i]
        b = vectorb[i]

        # Qubit 2*i (cos)
        qubit_key_cos = f"qubit {2*i}"
        results[qubit_key_cos] = []

        cos_results = {}
        cos_results["identifier"] = "cos"
        cos_results["state 0"] = abs((a + b) / np.sqrt(2))**2
        cos_results["state 1"] = abs((a - b) / np.sqrt(2))**2
        results[qubit_key_cos].append(cos_results)

        qubit_key_sin = f"qubit {2*i + 1}"
        results[qubit_key_sin] = []

        sin_results = {}
        sin_results["identifier"] = "sin"
        sin_results["state 0"] = abs((a + 1j*b) / np.sqrt(2))**2
        sin_results["state 1"] = abs((a - 1j*b) / np.sqrt(2))**2
        results[qubit_key_sin].append(sin_results)

    return json.dumps(results, indent=4)  

def run_quantum_simulation(vectora_q, vectorb_q):
    """
    Creates a quantum circuit, initializes qubits, applies gates,
    runs the circuit on a simulator, and prints the results.
    """

    if len(vectora_q) != len(vectorb_q):
        raise ValueError("Vectors must have the same length.")

    num_qubits = 2 * len(vectora_q)  # 2 qubits for each element
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Initialize qubits
    for i in range(len(vectora_q)):
        a = vectora_q[i]
        b = vectorb_q[i]

        # Qubit 2*i
        initial_state_0 = [a, b]
        qc.initialize(initial_state_0, 2*i)

        # Qubit 2*i + 1
        initial_state_1 = [a, b] 
        qc.initialize(initial_state_1, 2*i + 1)

        # Apply gates 
        qc.s(2*i + 1)

    qc.h(range(num_qubits))  # Apply Hadamard to all qubits
    qc.measure(range(num_qubits), range(num_qubits))  

    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=10000)  
    result = job.result()
    counts = result.get_counts(qc)

    # Calculate probabilities from counts
    total_shots = sum(counts.values())
    experimental_probs = {}
    for bitstring, count in counts.items():
        experimental_probs[bitstring] = count / total_shots

    # Calculate the probabilities based on the bitstring values
    probabilities = {}
    for i in range(num_qubits):
        probabilities[f"q{i}_0"] = 0.0

    for bitstring, prob in experimental_probs.items():
        for i in range(num_qubits):
            if bitstring[num_qubits - 1 - i] == '0':  
                probabilities[f"q{i}_0"] += prob

    # Create the JSON output
    json_output = json.dumps(probabilities, indent=4)
    return json_output

def calculate_complex_cosine_similarity(vectora, vectorb, simulation_result_json):
    """
    Calculates the final complex result based on the input vectors and the
    simulation result (JSON string).
    """

    if len(vectora) != len(vectorb):
        raise ValueError("Vectors must have the same length.")

    simulation_result = json.loads(simulation_result_json)  
    N = len(vectora)  
    if len(simulation_result) != 2*N:
        raise ValueError("Simulation result length does not match expected number of qubits.")

    sum_real = 0.0
    sum_img = 0.0

    for i in range(len(vectora)):  # Iterate from i=0 to N/2 -1
        a = vectora[i]
        b = vectorb[i]
        c2_2i = abs(a)**2 + abs(b)**2
        c2_2i_plus_1 = abs(a)**2 + abs(b)**2 

        # Extract probabilities from simulation result
        prob_2i = simulation_result.get(f"q{2*i}_0")
        prob_2i_plus_1 = simulation_result.get(f"q{2*i+1}_0")

        if prob_2i is None or prob_2i_plus_1 is None:
            raise ValueError(f"Missing probability for qubit {2*i} or {2*i+1} in simulation result.")

        sum_real += c2_2i * (2 * prob_2i - 1) / N
        sum_img += c2_2i_plus_1 * (2 * prob_2i_plus_1 - 1) / N

    return sum_real - 1j * sum_img


# Define the parameters for the "dog" and "cat" example
# Amplitudes and phases for dog (a) and cat (b)

e10 = 0.4
e11 = np.sqrt(1-e10**2)
e20 = 0.5
e21 = np.sqrt(1-e20**2)
phi1 = np.pi / 4
phi2 = np.pi / 3
varphi1 = np.pi / 6
varphi2 = np.pi / 2

print("imaginary")
print(2*e10*e20*np.sin(varphi1-phi1)+2*e11*e21*np.sin(varphi2-phi2))

vectora = [e10*np.exp(1j*varphi1),e11*np.exp(1j*varphi2)]
vectorb = [e20*np.exp(1j*phi1),e21*np.exp(1j*phi2)]

print("vectora0: ", vectora[0])
print("vectora1: ", vectora[1])
print("vectorb0: ", vectorb[0])
print("vectorb1: ", vectorb[1])

length_vectora = np.linalg.norm(vectora)
print("Length of vectora:", length_vectora)
length_vectorb = np.linalg.norm(vectorb)
print("Length of vectorb:", length_vectorb)

complex_similarity_exact = calculate_complex_similarity_exact(vectora, vectorb)
print("Complex similarity exact: ",complex_similarity_exact)

vectora_q, vectorb_q = create_converted_vectors(vectora, vectorb)

print("vectora_q0: ", vectora_q[0])
print("vectora_q1: ", vectora_q[1])
print("vectorb_q0: ", vectorb_q[0])
print("vectorb_q1: ", vectorb_q[1])

theoretical_results = calculate_qubit_states_theoretical(vectora_q, vectorb_q)
print("Theoretical results: ",theoretical_results)

simulation_results = run_quantum_simulation(vectora_q, vectorb_q)
print("Simuation results: ",simulation_results)

comple_cosine_similarity = calculate_complex_cosine_similarity(vectora, vectorb, simulation_results)
print("Complex cosine similarity experimental: ",comple_cosine_similarity)

