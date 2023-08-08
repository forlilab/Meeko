#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko multiprocessing worker
#
import sys
import logging
import traceback
import platform
import time

if platform.system() == "Darwin":  # mac
    import multiprocess as multiprocessing
else:
    import multiprocessing


class ParallelWriter(multiprocessing.Process):
    # this class is a listener that retrieves data from the queue and writes to files
    def __init__(
        self, output, queue, n_workers, pipe
    ):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        # this class knows about how many multi-processing workers there are and where the pipe to the parent is
        self.n_workers = n_workers
        self.pipe = pipe
        self.num_files_written = 0

        self.output = output

        self.input_mol_skipped = 0
        self.input_mol_with_failure = 0
        self.nr_failures = 0

    def run(self):
        # method overload from parent class
        #
        # this is where the task of this class is performed
        #
        # each multiprocessing.Process class must have a "run" method which
        # is called by the initialization (see below) with start()
        #
        try:
            self.time0 = time.perf_counter()
            while True:
                # retrieve the next task from the queue
                next_task = self.queue.get()
                if next_task is None:
                    # if a poison pill is found, it means one of the workers quit
                    self.n_workers -= 1
                    logging.debug(
                        "Closing process. Remaining open processes: "
                        + str(self.n_workers)
                    )
                else:
                    is_valid, this_mol_had_failure, nr_f, output_pdbqts_info = next_task
                    for pdbqt_string, name, covLabel_suffix in output_pdbqts_info:
                        self.output(pdbqt_string, name, covLabel_suffix)
                        self.num_files_written += 1
                    self.input_mol_skipped += int(is_valid==False)
                    self.input_mol_with_failure += int(this_mol_had_failure)
                    self.nr_failures += nr_f

                    # print info about files and time remaining
                    self.total_runtime = time.perf_counter() - self.time0
                    sys.stdout.write("\r")
                    sys.stdout.write(
                        "{0} PDBQTS written. Writing {1:.0f} files/minute. Elapsed time {2:.0f} seconds.".format(
                            self.num_files_written,
                            self.num_files_written * 60 / self.total_runtime,
                            self.total_runtime,
                        )
                    )
                    sys.stdout.flush()
                if self.num_readers == 0:
                    # no workers left, no job to do
                    logging.info("Molecule processing completed")
                    self.close()
                    break
        except Exception as e:
            tb = traceback.format_exc()
            self.pipe.send(
                (
                    e,
                    tb
                )
            )