[![License](https://img.shields.io/badge/license-%20GPLv3-blue.svg)](../master/LICENSE)

# This is an attempt to write an efficient SCF KAIN solver for Multiwavelets

This implementation makes use of the [VAMPyR](https://github.com/MRChemSoft/vampyr) library. Parts of the implementation are based on the SCF solver in [this](https://github.com/Dheasra/response) repository.

# Running the Code

This SCF solver makes use of the inital guess from an [MRChem](https://github.com/MRChemSoft/mrchem) calculation by loading the orbitals from MRChem. Therefore, it is important that both VAMPyR and MRChem are compiled useing the same version of the [MRCPP](https://github.com/MRChemSoft/mrcpp) library!
