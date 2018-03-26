# BPSAR-GPU
The Backprojection (Beam Formation) Imaging Algorithm for Synthetic Aperture Radar based on CUDA acceleration.

===========INTRODUCATION===========

The backprojection (BP) or so called Beam Formation (BF) Imaging Algorithm is designed in this project. The BP algorithm firstly is stated in the CT area, then the same idea is used in the synthetic aperture radar (SAR)

The Key idea of BP algorithm is coherently calculating the contribution of each pulse to each pixels, which, in details, Duersch PhD dissertation [1] or Jakowatz's paper [2]. Comparing to the well known Range-Doppler algorithm for ISAR/SAR, the most important advantage of the BP algorithm is non-distortion in wide angle, in other word, extremely high cross range resolution. However, the BP algorithm is extremely time-consumption. Even though several BP algorithm based on the sub-aperature algorithm by UCBerkely, the BPA is still to slow for very large scene.

Thus, in this project, I designed the BPA algorithm based on the CUDA acceleration.




Reference

[1] Duersch, M. I. (2013). Backprojection for synthetic aperture radar. Brigham Young University.

[2] Jakowatz, C. V., Wahl, D. E., & Yocky, D. A. (2008, April). Beamforming as a foundation for spotlight-mode SAR image formation by backprojection. In Algorithms for Synthetic Aperture Radar Imagery XV (Vol. 6970, p. 69700Q). International Society for Optics and Photonics.
