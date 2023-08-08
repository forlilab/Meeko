#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko multiprocessing manager
#

import platform
from time import sleep
import logging
import queue
import traceback
from meeko import ParallelWorker
from meeko import ParallelWriter
import sys

if platform.system() == "Darwin":  # mac
    import multiprocess as multiprocessing
else:
    import multiprocessing


class Parallelizer:
    def __init__(self, max_proc, mol_supplier, args, output, backend, is_covalent, preparator, covalent_builder) -> None:
        self.processed_mols = 0
        self.num_mols = 0
        self.n_workers = max_proc - 1  # reserve one core for pdbqt writing
        self.mol_supplier = mol_supplier
        self.queueIn = multiprocessing.Queue(maxsize=2 * max_proc)
        self.queueOut = multiprocessing.Queue(maxsize=2 * max_proc)

        self.args = args
        self.output = output
        self.backend = backend
        self.is_covalent = is_covalent
        self.preparator = preparator
        self.covalent_builder = covalent_builder

    def process_mols(self):
        self.workers = []
        self.p_conn, self.c_conn = multiprocessing.Pipe(True)
        logging.info("Starting {0} file readers".format(self.n_workers))

        for i in range(self.n_workers):
            s = ParallelWorker(self.args, self.output, self.backend, self.is_covalent, self.preparator, self.covalent_builder, self.queueIn, self.queueOut, self.c_conn)
            s.start()
            self.workers.append(s)
        
        w = ParallelWriter(self.output, self.queueOut, self.n_workers, self.c_conn)
        w.start()
        self.workers.append(w)

        # process items in the queue
        try:
            for mol in self.mol_supplier:
                self._add_to_queue(mol)
        except Exception as e:
            tb = traceback.format_exc()
            self._kill_all_workers("file sources", tb)
        # put as many poison pills in the queue as there are workers
        for i in range(self.num_readers):
            self.queueIn.put(None)

        # check for exceptions
        while w.is_alive():
            sleep(0.5)
            self._check_for_worker_exceptions()
        w.join()

        return w.input_mol_skipped, w.input_mol_with_failure, w.nr_failures

    def _add_to_queue(self, mol):
        max_attempts = 750
        timeout = 0.5  # seconds
        attempts = 0
        while True:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Something is blocking the multiprocessing queue. Exiting program."
                ) from queue.Full
            try:
                self.queueIn.put(mol, block=True, timeout=timeout)
                self.num_mols += 1
                self._check_for_worker_exceptions()
                break
            except queue.Full:
                # logging.debug(f"Queue full: queueIn.put attempt {attempts} timed out. {max_attempts - attempts} put attempts remaining.")
                attempts += 1
                self._check_for_worker_exceptions()

    def _check_for_worker_exceptions(self):
        if self.p_conn.poll():
            logging.debug("Caught error in multiprocessing")
            error, tb = self.p_conn.recv()
            self._kill_all_workers(error, tb)

    def _kill_all_workers(self, error, tb):
        for s in self.workers:
            s.kill()
        logging.debug(tb)
        raise error