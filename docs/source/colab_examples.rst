.. _colab_examples:

Colab Examples
============

`Google Colaboratory <https://colab.google/>`_ (Colab) is a cloud-based platform that allows users to write and execute Python codes through a browser. Regardless of the user's operating system, Colab provides Linux computing backends and some free GPU access. 

The following Colab examples are created to provide **an install-free experience** & **some generalizable workflows** of AutoDock Vina via Google Colab notebooks, which work in a similar manner to `Jupyter Notebooks <https://jupyter.org/>`_, in the pre-configured environment with `Meeko <https://github.com/forlilab/Meeko>`_ for receptor and ligand preparation, and other modules - `RDKit <https://rdkit.org/>`_, `Scrubber <https://github.com/forlilab/scrubber>`_, `ProDy <http://www.bahargroup.org/prody/>`_, `reduce2 <https://github.com/cctbx/cctbx_project/tree/master/mmtbx/reduce#reduce2>`_ (formerly `reduce <https://github.com/rlabduke/reduce>`_), and `py3Dmol <https://github.com/avirshup/py3dmol>`_ - for conformer generation, manipulation & pre-processing of protein structures and visualization. 

**Subscription is NOT required to run these Colab examples.** Additionally, the input files for the docking calculations are either directly pulled from open databases or generated from user inputs. With that, one can easily customize the notebooks and reuse the workflow for similar calculations on different biochemical systems. 

Overview
------------------------

**General Workflow of Docking Calculations in Examples**

.. image:: images/docking_workflow.png
   :alt: docking workflow
   :width: 100%
   :align: center

*Major Python packages used* 

* **RDKit** `https://rdkit.org/ <https://rdkit.org/>`_ 
* **Scrubber** `https://github.com/forlilab/scrubber <https://github.com/forlilab/scrubber>`_ 
* **Meeko** `https://github.com/forlilab/Meeko <https://github.com/forlilab/Meeko>`_ 
* **ProDy** `http://www.bahargroup.org/prody/ <http://www.bahargroup.org/prody/>`_ 
* **cctbx-base** (for reduce2) `https://github.com/cctbx/cctbx_project <https://github.com/cctbx/cctbx_project>`_ 
* **py3Dmol** `https://3dmol.org/ <https://3dmol.org/>`_ 

*Data* 

* **Phenix-project/geostd** (for reduce2) `https://github.com/phenix-project/geostd/ <https://github.com/phenix-project/geostd/>`_ 

[Scrubber -> Meeko -> Vina] Basic docking
------------------------

`Run on Colab! <https://colab.research.google.com/drive/1cHSl78lBPUc_J1IZxLgN4GwD_ADmohVU?usp=sharing>`_

The **basic docking example** is a rewrite based on the original basic docking example in the `Vina documentation <https://autodock-vina.readthedocs.io/en/latest/>`_. In this example, a small molecule ligand (Imatinib, PDB token `STI <https://www1.rcsb.org/ligand/STI>`_) is docked back to a hollow protein structure of mouse c-Abl (PDB token `1IEP <https://www1.rcsb.org/structure/1IEP>`_) to reproduce the complex structure. A docked pose that closely resembles the original position of the ligand is expected among the top-ranked poses. 


[Scrubber -> Meeko -> Vina] Flexible docking
------

`Run on Colab! <https://colab.research.google.com/drive/1cazEckGbvl9huWzpxXpd_Qaj0_NipWcz?usp=sharing>`_

The **flexible docking example** is a rewrite based on the original flexible docking example in the `Vina documentation <https://autodock-vina.readthedocs.io/en/latest/>`_. In this example, a variant of Imatinib (PDB token `PRC <https://www1.rcsb.org/ligand/PRC>`_) is docked back to a hollow protein structure of mouse c-Abl (PDB token `1FPU <https://www1.rcsb.org/structure/1FPU>`_) to reproduce the complex structure. Additionally, Thr315 is set to be a flexible residue. A docked pose that closely resembles the original position of the ligand and **a flipped Thr315** are expected among the top-ranked poses. 


[Scrubber -> Meeko -> Vina] Using AD4SF in Vina
---------------

`Run on Colab! <https://colab.research.google.com/drive/1zoSyID2fSoqGz3Zb1_IatUT2uxZ2mCNZ?usp=sharing>`_

The **using AutoDock4 (AD4) scoring function (SF) example** is a rewrite based on the corresponding part of the original basic docking example in the `Vina documentation <https://autodock-vina.readthedocs.io/en/latest/>`_. This example conducts the same redocking experiment as in *Basic docking* with the AutoDock4 scoring function instead of Vina. To do this, Autogrid4 is used to compute the grid maps, as an additional step after receptor preparation. 


[Scrubber -> Meeko -> Vina] Basic docking with an RNA receptor
---------------

`Run on Colab! <https://colab.research.google.com/drive/1hkt-XYebvAvbAf3cxZ3Yfze5R2lzhUfO?usp=sharing>`_

The basic docking example is developed after the **implementation of chemical templates for common nucleotides** in Meeko, which enables the preparation of RNA/DNA receptors. In this example, a small molecule inhibitor (Ribocil B, PDB token 51B) is docked back to a hollow protein structure of a bacteria FMN riboswitch (PDB token 5C45) to reproduce the complex structure.


[Scrubber -> Meeko -> Vina] Basic Docking with cofactors
---------------

`Run on Colab! <https://colab.research.google.com/drive/1-2yoPRVsmrLoYzLQGi_3KGWLlRk7ETdn?usp=sharing>`_

The basic docking example is developed to showcase the usage of **import additional chemical templates** into Meeko. In this example, a small molecule antibiotic (Kanamycin A, PDB token KAN) is docked back to a hollow protein structure of a bacteria aminoglycoside kinase APH(2)-Ia (PDB token 5IQB), together with two metal cofactor Magnesium (Mg2+) ions and the substrate phosphoaminophosphonic acid-guanylate ester (GMPPNP, PDB token GNP) to reproduce the complex structure.


[Scrubber -> Meeko -> AutoDock-GPU] Covalent Docking
---------------

`Run on Colab! <https://colab.research.google.com/drive/1tf9xOgn6u8eDTeFJtc8GCEGRX-8aR9Bo?usp=sharing>`_

The covalent docking example is based on the ***two-point attractor and flexible side chain method***. In this example, a small molecule substrate (Adenosine monophosphate, PDB token AMP) is attached onto the catalytic histidine residue of a hollow protein structure of bacteria RNA 3' cyclase (PDB token 3KGD) to reproduce the covalent intermediate complex structure. A docked pose that closely resembles the original position of the ligand is expected among the top-ranked poses.


[Scrubber -> Meeko -> AutoDock-GPU] Reactive Docking
---------------

`Run on Colab! <https://colab.research.google.com/drive/1tzQoguVQDCguOaLSsGvQuL57ry_PY3UG?usp=sharing>`_

The reactive docking example is based on reactive docking method that has been developed for high-throughput virtual screenings of reactive species. In this example, a small molecule substrate (Adenosine monophosphate, PDB token AMP) is targeting at the catalytic histidine residue of a hollow protein structure of bacteria RNA 3' cyclase (PDB token 3KGD) to generate the near-attack conformation for the formation of the phosphoamide bond. A docked pose that closely resembles the original position of the ligand is expected among the top-ranked poses. 


