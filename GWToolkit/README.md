![Tests](https://github.com/iphysresearch/GWToolkit/workflows/Tests/badge.svg) [![codecov](https://codecov.io/gh/iphysresearch/GWToolkit/branch/main/graph/badge.svg?token=43LD9IATDH)](https://codecov.io/gh/iphysresearch/GWToolkit)

# GWToolkit

GWToolkit: A Python Toolbox for gravitational wave astronomy.

## Installation

- Install [Conda](https://docs.conda.io/en/latest/miniconda.html)

<!-- ### Pre-requirements
 -->

### Packages used

- numpy
- pandas
- scipy
- scikit-learn
- corner
- lalsuite
- pycbc
- bilby
- astropy
- gwpy
- gwsurrogate
- ray

For details see `./environment.yml`

<!-- optional: -->

## Documentation

Documenation is available at: *.readthedocs.io (TBA)

## Contributing


Please see the guidelines:

Make sure conda/miniconda is installed in the operating system.

1. Clone this repo and change directory

  ```bash
  git clone git@github.com:iphysresearch/GWToolkit.git
  cd GWToolkit
  ```

2. Create conda environment

  ```bash
  make setup
  # or
  conda env create -f environment.yml
  ```

3. Activate the virtual environment

  ```bash
  conda activate gwtoolkit
  ```

  You must be very careful activating environments and always must remember to deactivate an environment once you are not going to use it because environments can be stacked.

  ```bash
  conda deactivate
  ```

4. Run the suite of tests and checks

  ```bash
  make check
  ```

5. Programming in jupyter lab/notebook

  ```bash
  # First, register the conda environment to Jupyter
  make add-to-jupyter
  # You can also remove the conda environment from Jupyter
  make remove-from-jupyter
  ```

  Now you can run jupyter lab/notebook and select `gwtoolkit` kernel:

  ```bash
  make lab
  # or 
  make notebook
  ```

  On top of your code, you can insert the following codes for importing

  ```python
  import sys, os
  sys.path.append('..')
  ```

---

Useful resources for contributing:

- How to Set up Python project
  - [How to Set Up a Python Project For Automation and Collaboration](https://eugeneyan.com/writing/setting-up-python-project-for-automation-and-collaboration/)
  - [Structuring Your Project](https://docs.python-guide.org/writing/structure/)
  - [Packaging a python library](https://blog.ionelmc.ro/2014/05/25/python-packaging/)
  - [The definitive guide to Python virtual environments with conda](https://whiteboxml.com/blog/the-definitive-guide-to-python-virtual-environments-with-conda)
  - [How to work with conda environments in shell scripts and Makefiles](https://blog.ianpreston.ca/conda/python/bash/2020/05/13/conda_envs.html)
  - [https://github.com/drkjam/condatools/](https://github.com/drkjam/condatools/)
  - [Create Virtualenv for Data Science Projects with One Command only](https://towardsdatascience.com/create-virtualenv-for-data-science-projects-with-one-command-only-7bec3548419f)
  - [花了两天，终于把 Python 的 setup.py 给整明白了](https://zhuanlan.zhihu.com/p/276461821?utm_oi=948852089393336320)
- Referencing Python Library
  - [https://github.com/scipy/scipy](https://github.com/scipy/scipy)
  - [https://github.com/lscsoft/bilby](https://github.com/lscsoft/bilby)
  - [https://github.com/matteobreschi/bajes](https://github.com/matteobreschi/bajes)
  - [https://github.com/sxs-collaboration/gwsurrogate](https://github.com/sxs-collaboration/gwsurrogate)
  - [https://gitlab.com/sborhanian/gwbench](https://gitlab.com/sborhanian/gwbench)
  - [https://github.com/stephengreen/lfi-gw](https://github.com/stephengreen/lfi-gw)
  - [https://github.com/timothygebhard/ggwd](https://github.com/timothygebhard/ggwd)
  - [https://github.com/mj-will/nessai](https://github.com/mj-will/nessai)
  - [https://github.com/pwwang/diot](https://github.com/pwwang/diot)
  - [https://github.com/kakiichi/lyaf-project](https://github.com/kakiichi/lyaf-project/blob/c7f92aa6d0b67e3c6d4b8deae4bc38f4340623ff/python_code/survey_requirement/cosmo_toolkits.py)
  - [https://vald-phoenix.github.io/pylint-errors/](https://vald-phoenix.github.io/pylint-errors/)
  - [https://github.com/pesummary/pesummary](https://github.com/pesummary/pesummary/blob/a7aeb8e587f5b378b641e963acbdab51d7d03566/pesummary/gw/file/conversions.py)
  - [https://github.com/qbaghi/bayesdawn](https://github.com/qbaghi/bayesdawn)
  - [https://github.com/sundayebosele/gwcelery](https://github.com/sundayebosele/gwcelery/blob/05e0b38789d185599046b22dcee5dbf97eb9c37f/gwcelery/tasks/lalinference.py)
  - [https://github.com/SSingh087/DL-estimating-parameters-of-GWs](https://github.com/SSingh087/DL-estimating-parameters-of-GWs)
- Gravitational waves
  - [A Thesaurus for Common Priors in Gravitational-Wave Astronomy (DCC-P2100112)](https://dcc.ligo.org/P2100112/)

<!-- Matplotlib 的默认颜色 以及 使用调色盘调整颜色
https://www.cnblogs.com/Gelthin2017/p/14177100.html -->


## Acknowledgements

TBA

## Citation

```bib
@software{


}

@article{

}
```