./wet.py -i ligand.pdbqt
# edit the GPF before this step...
autogrid4 -p protein.gpf -l protein.glg
./mapwater.py  -r protein.pdbqt -s protein.W.map
# edit the DPF before this step...
autodock4 -p ligand_HYDRO_protein.dpf -l ligand_HYDRO_protein.dlg
./dry.py -c -r protein.pdbqt -c -m protein.W.map -i ligand_HYDRO_protein.dlg


