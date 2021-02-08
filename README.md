# Bilinear Dynamic Mode Decomposition for Quantum Control

https://arxiv.org/abs/2010.14577

Andy Goldschmidt, Eurika Kaiser, Jonathan L. Dubois, Steven L. Brunton, J. Nathan Kutz

Data-driven methods for establishing quantum optimal control (QOC) using time-dependent control pulses tailored to specific quantum dynamical systems and desired control objectives are critical for many emerging quantum technologies. We develop a data-driven regression procedure, bilinear dynamic mode decomposition (biDMD), that leverages time-series measurements to establish quantum system identification for QOC. The biDMD optimization framework is a physics-informed regression that makes use of the known underlying Hamiltonian structure. Further, the biDMD can be modified to model both fast and slow sampling of control signals, the latter by way of stroboscopic sampling strategies. The biDMD method provides a flexible, interpretable, and adaptive regression framework for real-time, online implementation in quantum systems. Further, the method has strong theoretical connections to Koopman theory, which approximates non-linear dynamics with linear operators. In comparison with many machine learning paradigms, it requires minimal data and the biDMD model is easily updated as new data is collected. We demonstrate the efficacy and performance of the approach on a number of representative quantum systems, showing that it also matches experimental results.
