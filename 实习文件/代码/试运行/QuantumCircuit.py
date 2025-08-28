from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

m, n = 1, 1
# 4 比特：0,1=辅助贝尔态；2,3=数据贝尔态
qc = QuantumCircuit(4)

# 1) 造辅助 Bell 态 (q0,q1)
qc.h(0)
qc.cx(0,1)

# 2) 造数据 Bell 态 (q2,q3)
qc.h(2)
qc.cx(2,3)

# 3) U_p 作用在 “后1辅助+前1数据” = q1,q2
U_p = efficient_su2(2, reps=1, entanglement='linear')
qc.append(U_p, [1,2])
print(qc.draw())