#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko multiprocessing worker
#
import sys
import logging
import traceback
import platform
import queue
from .preparation import MoleculePreparation

if platform.system() == "Darwin":  # mac
    import multiprocess as multiprocessing
else:
    import multiprocessing

class ParallelWorker(multiprocessing.Process):
    def __init__(self, args, output, backend, is_covalent, preparator, covalent_builder, queueIn, queueOut, pipe_conn) -> None:
        self.args = args
        self.output = output
        self.backend = backend
        self.is_covalent = is_covalent
        self.preparator = preparator
        self.covalent_builder = covalent_builder

        # initialize the parent class to inherit all multiprocessing methods
        multiprocessing.Process.__init__(self)
        # each worker knows the queue in (where data to process comes from)
        self.queueIn = queueIn
        # ...and a queue out (where to send the results)
        self.queueOut = queueOut
        # ...and a pipe to the parent
        self.pipe = pipe_conn

    def run(self):
        try:
            while True:
                # retrieve from the queue in the next task to be done
                next_task = self.queueIn.get()
                # if a poison pill is received, this worker's job is done, quit
                if next_task is None:
                    # before leaving, pass the poison pill back in the queue
                    self.queueOut.put(None)
                    break

                # generate CPU LOAD
                output_data = MoleculePreparation.prep_single_mol(next_task, self.args, self.output, self.backend, self.is_covalent, self.preparator, self.covalent_builder, write_output=False)
                self.queueOut.put(output_data)

        except Exception as e:
            tb = traceback.format_exc()
            self.pipe.send(
                (e, tb)
            )
        finally:
            return
        
    def _add_to_queueout(self, obj):
        max_attempts = 750
        timeout = 0.5  # seconds
        attempts = 0
        while True:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Something is blocking the output queue. Exiting program."
                ) from queue.Full
            try:
                self.queueOut.put(obj, block=True, timeout=timeout)
                break
            except queue.Full:
                # logging.debug(f"Queue full: queueOut.put attempt {attempts} timed out. {max_attempts - attempts} put attempts remaining.")
                attempts += 1
