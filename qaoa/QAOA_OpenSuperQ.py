from qiskit import *
from qiskit import qobj as qiskit_qobj
from qiskit.providers import BaseBackend, BaseJob, BaseProvider
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.providers.providerutils import filter_backends
from qiskit.transpiler import PassManager
from qiskit.result import Result
import  numpy as np
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua import QuantumInstance
from qiskit.aqua.translators.ising import max_cut
from qiskit.aqua.components.optimizers import COBYLA, POWELL
from qiskit.quantum_info import Pauli

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.visualization import circuit_drawer


#IBMQ.save_account('294d92aed2a23d0c44832f579c2f091e7a37da02065d4e5ddde0db7397c7eb3cf94a201b71bc404a4370944218cb9ee5a40bfea69aff930432224d055119e6cd', overwrite=True)
IBMQ.load_account()
#tmp.gates


def run_ETH_Simulator(number_of_qubits, list_of_qubits, shots):

    hadamard_list = [0] * number_of_qubits
    for qubit in list_of_qubits:
        hadamard_list[qubit] = (1 + hadamard_list[qubit]) % 2

    # Calculate the result for each basis state
    result = [0] * (2 ** number_of_qubits)
    for i in range(2 ** number_of_qubits):
        # Example: when i is 2,
        # the basis_state is 01000
        basis_state = '{0:b}'.format(i).zfill(number_of_qubits)[::-1]

        for qubit in range(number_of_qubits):
            if hadamard_list[qubit] == 0 and basis_state[qubit] == '1':
                result[i] = 0
                break
            if hadamard_list[qubit] == 1:
                result[i] += int(shots / (2 ** (1 + hadamard_list.count(1))))

    return result


class OpenSuperQ_Job(BaseJob):
    def __init__(self, backend):
        super().__init__(backend, 1)

    def result(self,timeout=None):
        return self._result

    def cancel(self):
        pass

    def status(self):
        pass

    def submit(self):
        pass


class ETH_7_rev_1_Backend(BaseBackend):
    '''
    A wrapper backend for the ETH 7 rev 1 chip
    '''

    def __init__(self, provider=None):
        configuration = {
            'backend_name': 'OSQ_ETH7_rev1',
            'description': 'OpenSuperQ ETH 7 qubit chip, rev. 1, 12_15702',
            'backend_version': '0.1.0',
            'url': 'http://opensuperq.eu/',
            'sample_name': 'QUDEV_M_12_15702',
            'n_qubits': 7,
            'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id'],
            'coupling_map': [[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [3, 5], [3, 6], [4, 6],[2, 0], [3, 0], [3, 1], [4, 1], [5, 2], [5, 3], [6, 3], [6, 4]],
            # Reduced qubit numbers by 1 compared to 20190823_OSQ_Waypoint1_experimental_parameters.pdf
            'simulator': True,
            'local': True,
            'open_pulse': False,
            'conditional': False,
            'n_registers': 1,  # Really 0, but QISKIT would not allow it, even if 'conditional' is False
            'max_shots': 1_000_000,
            'memory': 0,
            'credits_required': False,
            'gates': [
                    {'name':'id',
                    'parameters':[],
                    'coupling_map':[[0], [1], [2], [3], [4], [5], [6]],
                     'qasm_def' : 'gate id q { U(0,0,0) q; }'},

                    {'name':'u1',
                    'parameters':['lambda'],
                    'coupling_map':[[0], [1], [2], [3], [4], [5], [6]],
                    'qasm_def':'gate u1(lambda) q { U(0,0,lambda) q; }'},

                    {'name':'u2',
                    'parameters':['phi', 'lambda'],
                    'coupling_map':[[0], [1], [2], [3], [4], [5], [6]],
                    'qasm_def':'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'},

                    {'name':'u3',
                    'parameters':['theta', 'phi', 'lambda'],
                    'coupling_map':[[0], [1], [2], [3], [4], [5], [6]],
                    'qasm_def':'u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'},

                    {'name':'cx',
                    'parameters':[],
                    'coupling_map':[[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [3, 5], [3, 6], [4, 6],[2, 0], [3, 0], [3, 1], [4, 1], [5, 2], [5, 3], [6, 3], [6, 4]],
                    'qasm_def':'gate cx q1,q2 { CX q1,q2; }'}
            ]
        }

        super().__init__(configuration=BackendConfiguration.from_dict(configuration),
                         provider=provider)

        self.aer_simulator = Aer.get_backend('qasm_simulator')



    def run(self, qobj):

        for circuit_index, circuit in enumerate(qobj.experiments):
            for operation in circuit.instructions:
                if getattr(operation, 'conditional', None):
                    raise QiskitError('conditional operations are not supported '
                                      'by the Hadamard simulator')
        # Execute and get counts
        #simulation_results = execute(qobj, self.aer_simulator).result()

        job = OpenSuperQ_Job(None)

        experiment_results = []
        for circuit_index, circuit in enumerate(qobj.experiments):
            number_of_qubits = circuit.config.n_qubits
            shots = qobj.config.shots

            list_of_qubits = []
            list_of_qubits.append(operation.qubits[0])

            counts = run_ETH_Simulator(number_of_qubits, list_of_qubits, shots)

            formatted_counts = {}
            for i in range(2 ** number_of_qubits):
                if counts[i] != 0:
                    formatted_counts[hex(i)] = counts[i]

            experiment_results.append({
                'name': circuit.header.name,
                'success': True,
                'shots': shots,
                'data': {'counts': formatted_counts},
                'header': circuit.header.to_dict()
            })
        job._result = Result.from_dict({
            'results': experiment_results,
            'backend_name': 'test',
            'backend_version': '0.1.0',
            'qobj_id': '0',
            'job_id': '0',
            'success': True
        })

        return job


class OpenSuperQ_Provider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of Hadamard backends
        self._backends = [ETH_7_rev_1_Backend(provider=self)]


    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [backend for backend in backends if backend.name() == name]

        return filter_backends(backends, filters=filters, **kwargs)





def get_ising_qubitops(isng,qubits):
    h_i = {}
    j_ij = {}
    constant = 0
    for (key, value) in zip(isng.keys(), isng.values()):
        if len(key) == 1:
          h_i[key[0]] = value
        elif len(key) == 2:
          j_ij[key] = value
        elif len(key) == 0:
                constant = value

    num_nodes = qubits
    pauli_list = []

    for key, value in j_ij.items():
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[key[0]] = True
                zp[key[1]] = True
                pauli_list.append([value, Pauli(zp, xp)])
    for key, value in h_i.items():
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[key] = True
                pauli_list.append([value, Pauli(zp, xp)])
    return WeightedPauliOperator(paulis=pauli_list), constant


isingdict ={(0,1):0.5,(1,2):0.5,(2,3):0.5,(3,4):0.5,():-2.0}
qubitOp, shift  = get_ising_qubitops(isingdict,5)

provider = OpenSuperQ_Provider()
backend = provider.get_backend('OSQ_ETH7_rev1')

optimizer = POWELL()

qaoa = QAOA(qubitOp, optimizer, p=1, operator_mode='paulis',initial_point=[0.0, 0.0])
quantum_instance = QuantumInstance(backend)

circ = qaoa.construct_circuit(parameter=[2.0, 1.0],circuit_name_prefix='Hello',statevector_mode=True)
latex = circ[0].draw(output='latex_source')

result = qaoa.run(quantum_instance)

print('shift: ', shift)
print('name: ', result.keys())
print([str(key) + " " + str(value) for key, value in result.items()])

plot_histogram(result['eigvecs'], figsize = (18,5))
x = max_cut.sample_most_likely(result['eigvecs'][0])
graph_solution = max_cut.get_graph_solution(x)
#qaoa._circuit.draw(output='mpl')

print('Hello')
