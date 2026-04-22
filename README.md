# simPle
Simulation of Planet evolution that tracks the chemical composition of the material accreted onto the protoplanet. 
It is based on a 1D model for the advection and diffusion of molecules through protoplanetary discs, drift_composition [issues page](https://github.com/rbooth200/drift_composition/). 

Installation
------------
You can install the code with `pip`
```
python -m pip install .
```
If you want to be able to edit the source file you may find it convenient to pass the `-e` flag to `pip`:
```
python -m pip install -e .
```


Getting started
---------------
To see how the code can be used to compute the composition of a protoplanetary disc see the tutorial in `examples/basic_model.ipynb`
For an example of the compositional evolution of a planet see the tutorial in `examples/planet_example.ipynb`

Authors
-------
 - Richard Booth (University of Leeds) <https://github.com/rbooth200>
 - Anna Penzlin (LMU Munich) <https://github.com/miosta>
   
Contact
-------
Submit a bug report or feature request on the [issues page](https://github.com/miosta/simPle/issues). For anything else, feel free to [email me](mailto:A.Penzlin@lmu.de).

License
-------
This is free software licensed under the GPLv3 License. For more details see the [LICENSE](https://github.com/rbooth200/drift_composition//blob/master/LICENSE).

© Copyright 2023 Richard Booth
