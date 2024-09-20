"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from numpy import append
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import utcnow, DLIOMPI

import os
import json
import math
import logging
import pandas as pd
from time import time
import numpy as np
import torch

import socket
class StatsCounter(object):

    def __init__(self):
        self.comm = DLIOMPI.get_instance().comm()
        self.args = ConfigArguments.get_instance()
        self.my_rank = self.args.my_rank
        self.comm_size = self.args.comm_size
        self.output_folder = self.args.output_folder
        self.record_size = self.args.record_length
        self.batch_size = self.args.batch_size
        self.batch_size_eval = self.args.batch_size_eval
        self.summary = {}
        self.summary['start'] = utcnow()
        self.summary['num_accelerators'] = self.comm_size
        self.summary['hostname'] = socket.gethostname()
        self.summary['metric'] = {}
        self.summary['num_files_train'] = self.args.num_files_train
        self.summary['num_files_eval'] = self.args.num_files_eval
        self.summary['num_samples_per_file'] = self.args.num_samples_per_file
        max_steps = math.floor(self.args.num_samples_per_file * self.args.num_files_train / self.args.batch_size / self.args.comm_size)

        if self.args.total_training_steps > 0:
            if self.args.total_training_steps > max_steps:
                logging.error(f"Only have enough data for {max_steps} steps but {self.args.total_training_steps} wanted")
                exit(-1)
            self.steps_override = True
            self.steps = self.args.total_training_steps
        else:
            self.steps_override = False
            self.steps = max_steps
        
        self.steps_eval = math.floor(self.args.num_samples_per_file * self.args.num_files_eval / self.args.batch_size_eval / self.args.comm_size)
        # Only the root process keeps track of overall stats
        if self.my_rank == 0:
            self.per_epoch_stats = {}
        # Each process keeps track of its loading and processing times independently
        self.output = {}
        self.train_au = []
        self.eval_au = []
        self.train_throughput = []
        self.eval_throughput = []
        self.parameter_read = []
        self.parameter_write = []
        self.gradient_read = []
        self.gradient_write = []
        self.optimizer_read = []
        self.optimizer_write = []
    def start_run(self):
        self.start_run_timestamp = time()
    def end_run(self):
        self.end_run_timestamp = time()
        if not self.args.generate_only:
            total_elapsed_time = self.end_run_timestamp - self.start_run_timestamp
            train_au = np.array(self.comm.allreduce(np.array(self.train_au)))/self.comm.size
            train_throughput = self.comm.allreduce(np.array(self.train_throughput))
            self.summary['epochs'] = len(train_au)
            self.summary['metric']['train_au_percentage'] = list(train_au)
            self.summary['metric']['train_au_mean_percentage'] = np.mean(train_au)
            if self.summary['metric']['train_au_mean_percentage'] >=90:
                self.summary['metric']['train_au_meet_expectation'] = 'success'
            else:
                self.summary['metric']['train_au_meet_expectation'] = 'fail'
            self.summary['metric']['train_au_stdev_percentage'] = np.std(train_au)
            self.summary['metric']['train_throughput_samples_per_second'] = list(train_throughput)
            self.summary['metric']['train_throughput_mean_samples_per_second'] = np.mean(train_throughput)
            self.summary['metric']['train_throughput_stdev_samples_per_second'] = np.std(train_throughput)
            self.summary['metric']['train_io_mean_MB_per_second'] = np.mean(train_throughput)*self.record_size/1024./1024.
            self.summary['metric']['train_io_stdev_MB_per_second'] = np.std(train_throughput)*self.record_size/1024./1024.
            if self.args.ZeRO_Infinity:
                if self.args.parameter_swapping is True:
                    self.summary['metric']['ZeRO_Infinity_total_byte_of_parameter'] = f"{torch.tensor([], dtype=self.args.parameter_dtype).element_size()*self.args.epochs*self.args.parameter_size/(1024.*1024.)} MB"
                if self.args.gradient_swapping is True:
                    self.summary['metric']['ZeRO_Infinity_total_byte_of_gradient'] = f"{torch.tensor([], dtype=self.args.gradient_dtype).element_size()*self.args.epochs*self.args.gradient_size/(1024.*1024.)} MB"
                if self.args.optimizer_swapping is True:
                    self.summary['metric']['ZeRO_Infinity_total_byte_of_optimizer'] = f"{torch.tensor([], dtype=self.args.optimizer_dtype).element_size()*self.args.epochs*self.args.optimizer_size/(1024.*1024.)} MB"
            if self.args.do_eval:
                eval_au = np.array(self.comm.allreduce(self.eval_au))/self.comm.size
                eval_throughput = self.comm.allreduce(self.eval_throughput)
                self.summary['metric']['eval_au_percentage'] = list(eval_au)
                self.summary['metric']['eval_au_mean_percentage'] = np.mean(eval_au)
                if self.summary['metric']['eval_au_mean_percentage'] >=90:
                    self.summary['metric']['eval_au_meet_expectation'] = 'success'
                else:
                    self.summary['metric']['eval_au_meet_expectation'] = 'fail'
                self.summary['metric']['eval_au_stdev_percentage'] = np.std(eval_au)
                self.summary['metric']['eval_throughput_samples_per_second'] = list(eval_throughput)
                self.summary['metric']['eval_throughput_mean_samples_per_second'] = np.mean(eval_throughput)
                self.summary['metric']['eval_throughput_stdev_samples_per_second'] = np.std(eval_throughput)
                self.summary['metric']['eval_io_mean_MB_per_second'] = np.mean(eval_throughput)*self.record_size/1024./1024.
                self.summary['metric']['eval_io_stdev_MB_per_second'] = np.std(eval_throughput)*self.record_size/1024./1024.
            if self.my_rank==0:
                logging.info(f"{utcnow()} Saved outputs in {self.output_folder}")   
                metric="Averaged metric over all epochs\n[METRIC] ==========================================================\n"
                metric = metric + f"[METRIC] Training Accelerator Utilization [AU] (%): {np.mean(train_au):.4f} ({np.std(train_au):.4f})\n"
                metric = metric + f"[METRIC] Training Throughput (samples/second): {np.mean(train_throughput):.4f} ({np.std(train_throughput):.4f})\n"
                metric = metric + f"[METRIC] Training I/O Throughput (MB/second): {np.mean(train_throughput)*self.record_size/1024/1024:.4f} ({np.std(train_throughput)*self.record_size/1024/1024:.4f})\n"
                metric = metric + f"[METRIC] train_au_meet_expectation: {self.summary['metric']['train_au_meet_expectation']}\n"

                if self.args.do_eval:
                    metric = metric + f"[METRIC] Eval Accelerator Utilization [AU] (%): {np.mean(eval_au):.4f} ({np.std(eval_au):.4f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (samples/second): {np.mean(eval_throughput):.6f} ({np.std(eval_throughput):.6f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (MB/second): {np.mean(eval_throughput)*self.record_size/1024/1024:.6f} ({np.std(eval_throughput)*self.record_size/1024/1024:.6f})\n"
                    metric = metric + f"[METRIC] eval_au_meet_expectation: {self.summary['metric']['eval_au_meet_expectation']}\n"
                metric+="[METRIC] ==========================================================\n"
                logging.info(metric)   
    def end_run_ZeRO_Infinity(self):
        if not self.args.generate_only:
            parameter_byte = torch.tensor([], dtype=self.args.parameter_dtype).element_size()*self.args.epochs*self.args.parameter_size/(1024.*1024.)
            gradient_byte = torch.tensor([], dtype=self.args.gradient_dtype).element_size()*self.args.epochs*self.args.gradient_size/(1024.*1024.)
            optimizer_byte = torch.tensor([], dtype=self.args.optimizer_dtype).element_size()*self.args.epochs*self.args.optimizer_size/(1024.*1024.)
            if self.args.parameter_swapping is True:
                # logging.info(f"parameter read {len(self.parameter_read)}")
                # logging.info(f"parameter write {len(self.parameter_write)}")
                self.summary['metric']['ZeRO_Infinity_parameter_write_MB_per_second'] = (parameter_byte)/(np.sum(self.parameter_write))
                self.summary['metric']['ZeRO_Infinity_parameter_read_MB_per_second'] = (2*parameter_byte)/(np.sum(self.parameter_read))
            if self.args.gradient_swapping is True:
                # logging.info(f"gradient read {len(self.gradient_read)}")
                # logging.info(f"gradient write {len(self.gradient_write)}")
                self.summary['metric']['ZeRO_Infinity_gradient_write_MB_per_second'] = (gradient_byte)/(np.sum(self.gradient_write))
                self.summary['metric']['ZeRO_Infinity_gradient_read_MB_per_second'] = (gradient_byte)/(np.sum(self.gradient_read))
            if self.args.optimizer_swapping is True:
                # logging.info(f"optimizer read {len(self.optimizer_read)}")
                # logging.info(f"optimizer write {len(self.optimizer_write)}")
                self.summary['metric']['ZeRO_Infinity_optimizer_write_MB_per_second'] = (optimizer_byte)/(np.sum(self.optimizer_write))
                self.summary['metric']['ZeRO_Infinity_optimizer_read_MB_per_second'] = (optimizer_byte)/(np.sum(self.optimizer_read))

    def start_train(self, epoch):   
        if self.my_rank == 0:
            ts = utcnow()
            if self.steps_override:
                logging.info(f"{ts} Starting epoch {epoch}: Overriding number of steps to {self.steps}.")
            else:
                logging.info(f"{ts} Starting epoch {epoch}: {self.steps} steps expected")
            self.per_epoch_stats[epoch] = {
                'start': ts,
            }
        # Initialize dicts for the current epoch
        self.output[epoch] = {}
        self.output[epoch]['load'] = {}
        self.output[epoch]['proc'] = {}
        self.output[epoch]['throughput'] = {}
        self.output[epoch]['au'] = {}
        self.output[epoch]['compute'] = {}

    def start_ZeRO_Infinity(self, epoch, parameter_swapping, gradient_swapping, optimizer_swapping):
        if parameter_swapping is True:
            self.output[epoch]['parameter_read'] = {}
            self.output[epoch]['parameter_write'] = {}
        if gradient_swapping is True:
            self.output[epoch]['gradient_read'] = {}
            self.output[epoch]['gradient_write'] = {}
        if optimizer_swapping is True:
            self.output[epoch]['optimizer_read'] = {}
            self.output[epoch]['optimizer_write'] = {}

    def end_train(self, epoch, steps):
        au = np.array([self.output[epoch]['au'][k] for k in self.output[epoch]['au']])
        throughput = np.array([self.output[epoch]['throughput'][k] for k in self.output[epoch]['throughput']])
        steps = np.array([len(self.output[epoch]['proc'][k]) for k in self.output[epoch]['throughput']])
        if (np.sum(steps)==0):
            au = 0.0
            throughput = 0.0
        else:
            au = np.sum(au*steps)/np.sum(steps)
            throughput = np.sum(throughput*steps)/np.sum(steps)
        self.train_au.append(au)
        self.train_throughput.append(throughput)

        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch]['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            self.per_epoch_stats[epoch]['end'] = ts
            self.per_epoch_stats[epoch]['duration'] = duration
            logging.info(f"{ts} Ending epoch {epoch} - {np.sum(steps)} steps completed in {duration} s")

    def end_train_ZeRO_Infinity(self, epoch, steps):
        if self.args.parameter_swapping is True:
            ar = np.array([self.output[epoch]['parameter_read'][k] for k in self.output[epoch]['parameter_read']])
            added = np.sum(ar)
            self.parameter_read.append(added)
            ar = np.array([self.output[epoch]['parameter_write'][k] for k in self.output[epoch]['parameter_write']])
            added = np.sum(ar)
            self.parameter_write.append(added)
        if self.args.optimizer_swapping is True:
            ar = np.array([self.output[epoch]['optimizer_read'][k] for k in self.output[epoch]['optimizer_read']])
            added = np.sum(ar)
            self.optimizer_read.append(added)
            ar = np.array([self.output[epoch]['optimizer_write'][k] for k in self.output[epoch]['optimizer_write']])
            added = np.sum(ar)
            self.optimizer_write.append(added)
        if self.args.gradient_swapping is True:
            ar = np.array([self.output[epoch]['gradient_read'][k] for k in self.output[epoch]['gradient_read']])
            added = np.sum(ar)
            self.gradient_read.append(added)
            ar = np.array([self.output[epoch]['gradient_write'][k] for k in self.output[epoch]['gradient_write']])
            added = np.sum(ar)
            self.gradient_write.append(added)

    def start_eval(self, epoch):
        self.start_timestamp = time()
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting eval - {self.steps_eval} steps expected")
            self.per_epoch_stats[epoch]['eval'] = {
                'start': ts
            }
        self.output[epoch]['load']['eval'] = []
        self.output[epoch]['proc']['eval'] = []
        self.output[epoch]['compute']['eval'] = []
        self.output[epoch]['au']['eval'] = 0.0
        self.output[epoch]['throughput']['eval'] = 0.0
    def end_eval(self, epoch):
        self.end_timestamp = time()
        self.compute_metrics_eval(epoch)
        self.eval_au.append(self.output[epoch]['au']['eval'])
        self.eval_throughput.append(self.output[epoch]['throughput']['eval'] )
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts)- pd.to_datetime(self.per_epoch_stats[epoch]['eval']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending eval - {self.steps_eval} steps completed in {duration} s")
            self.per_epoch_stats[epoch]['eval']['end'] = ts
            self.per_epoch_stats[epoch]['eval']['duration'] = duration        
            logging.info(f"{utcnow()} Epoch {epoch} [Eval] Accelerator Utilization [AU] (%): {self.output[epoch]['au']['eval']:.4f}")
            logging.info(f"{utcnow()} Epoch {epoch} [Eval] Throughput (samples/second): {self.output[epoch]['throughput']['eval']*self.comm_size:.4f}")

    def start_block(self, epoch, block):
        self.start_timestamp = time()
        self.output[epoch]['load'][f'block{block}'] = []
        self.output[epoch]['proc'][f'block{block}'] = []
        self.output[epoch]['throughput'][f'block{block}'] = []
        self.output[epoch]['au'][f'block{block}'] = []
        self.output[epoch]['compute'][f'block{block}'] = []
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting block {block}")
            self.per_epoch_stats[epoch][f'block{block}'] = {
                'start': ts
            }

    def start_block_ZeRO_Infinity(self, epoch, block, args):
        if args.parameter_swapping is True:
            self.output[epoch]['parameter_read'][f'block{block}'] = []
            self.output[epoch]['parameter_write'][f'block{block}'] = []
        if args.gradient_swapping is True:
            self.output[epoch]['gradient_read'][f'block{block}'] = []
            self.output[epoch]['gradient_write'][f'block{block}'] = []
        if args.optimizer_swapping is True:
            self.output[epoch]['optimizer_read'][f'block{block}'] = []
            self.output[epoch]['optimizer_write'][f'block{block}'] = []

    def end_block(self, epoch, block, steps_taken):
        self.end_timestamp = time()
        self.compute_metrics_train(epoch, block)
        
        if self.my_rank == 0:
            # Block was possibly already ended. Need this to end blocks
            # still ongoing when data loader runs out of batches and
            # does not take one of the expected exits from the batch reading loop
            if 'end' in self.per_epoch_stats[epoch][f'block{block}']:
                return
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'block{block}']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending block {block} - {steps_taken} steps completed in {duration} s")
            self.per_epoch_stats[epoch][f'block{block}']['end'] = ts
            self.per_epoch_stats[epoch][f'block{block}']['duration'] = duration
            logging.info(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Accelerator Utilization [AU] (%): {self.output[epoch]['au'][f'block{block}']:.4f}")
            logging.info(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Throughput (samples/second): {self.output[epoch]['throughput'][f'block{block}']*self.comm_size:.4f}")

    def start_ckpt(self, epoch, block, steps_taken):
        if self.my_rank == 0:
            ts = utcnow()
            logging.info(f"{ts} Starting checkpoint {block} after total step {steps_taken} for epoch {epoch}")
            self.per_epoch_stats[epoch][f'ckpt{block}'] = {
                'start': ts
            }

    def end_ckpt(self, epoch, block):
        if self.my_rank == 0:
            ts = utcnow()
            duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'ckpt{block}']['start'])
            duration = '{:.2f}'.format(duration.total_seconds())
            logging.info(f"{ts} Ending checkpoint {block} for epoch {epoch}")

            self.per_epoch_stats[epoch][f'ckpt{block}']['end'] = ts
            self.per_epoch_stats[epoch][f'ckpt{block}']['duration'] = duration

    def batch_loaded(self, epoch, step, block, t0):
        duration = time() - t0
        key = f'block{block}'
        if key in self.output[epoch]['load']:
            self.output[epoch]['load'][key].append(duration)
        else:
            self.output[epoch]['load'][key] = [duration]
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step}: loaded {self.batch_size} samples in {duration} s")


    def batch_processed(self, epoch, step, block, t0, computation_time):
        duration = time() - t0
        key = f'block{block}'
        if key in self.output[epoch]['proc']:
            self.output[epoch]['proc'][key].append(duration)
            self.output[epoch]['compute'][key].append(computation_time)
        else:
            self.output[epoch]['proc'] = [duration]
            self.output[epoch]['compute']=[computation_time]
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size} samples in {duration} s")

    def ZeRO_Infinity_processed_parameter_read(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['parameter_read']:
            self.output[epoch]['parameter_read'][key].append(computation_time)
        else:
            self.output[epoch]['parameter_read'] = [computation_time]
    
    def ZeRO_Infinity_processed_parameter_write(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['parameter_write']:
            self.output[epoch]['parameter_write'][key].append(computation_time)
        else:
            self.output[epoch]['parameter_write'] = [computation_time]
    
    def ZeRO_Infinity_processed_gradient_read(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['gradient_read']:
            self.output[epoch]['gradient_read'][key].append(computation_time)
        else:
            self.output[epoch]['gradient_read'] = [computation_time]
    
    def ZeRO_Infinity_processed_gradient_write(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['gradient_write']:
            self.output[epoch]['gradient_write'][key].append(computation_time)
        else:
            self.output[epoch]['gradient_write'] = [computation_time]
    
    def ZeRO_Infinity_processed_optimizer_read(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['optimizer_read']:
            self.output[epoch]['optimizer_read'][key].append(computation_time)
        else:
            self.output[epoch]['optimizer_read'] = [computation_time]
    
    def ZeRO_Infinity_processed_optimizer_write(self, epoch, block, computation_time):
        key = f'block{block}'
        if key in self.output[epoch]['optimizer_write']:
            self.output[epoch]['optimizer_write'][key].append(computation_time)
        else:
            self.output[epoch]['optimizer_write'] = [computation_time]

    def compute_metrics_train(self, epoch, block):
        key = f"block{block}"
        total_compute_time = np.sum(self.output[epoch]['compute'][key][1:])
        if (total_compute_time==0):
            au=0.0
        else:
            total_time = self.end_timestamp - self.start_timestamp - self.output[epoch]['proc'][key][0]
            au = total_compute_time / total_time
        throughput = len(self.output[epoch]['compute'][key])/(self.end_timestamp - self.start_timestamp)*self.batch_size
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def compute_metrics_eval(self, epoch):
        key = 'eval'
        total_compute_time = np.sum(self.output[epoch]['compute'][key][1:])
        if (total_compute_time==0):
            au=0.0
        else:
            total_time = self.end_timestamp - self.start_timestamp - self.output[epoch]['proc'][key][0]
            au = total_compute_time / total_time
        throughput = len(self.output[epoch]['compute'][key])/(self.end_timestamp - self.start_timestamp)*self.batch_size_eval
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def eval_batch_loaded(self, epoch, step, t0):
        duration = time() - t0
        self.output[epoch]['load']['eval'].append(duration)
        logging.debug(f"{utcnow()} Rank {self.my_rank} step {step} loaded {self.batch_size_eval} samples in {duration} s")


    def eval_batch_processed(self, epoch, step, t0, computation_time):
        duration = time() - t0
        self.output[epoch]['proc']['eval'].append(duration)
        self.output[epoch]['compute']['eval'].append(computation_time)
        logging.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size_eval} samples in {duration} s")
    def finalize(self):
        self.summary['end'] = utcnow()
    def save_data(self):
        # Dump statistic counters to files for postprocessing
        # Overall stats
        if self.my_rank == 0:
            with open(os.path.join(self.output_folder, 'per_epoch_stats.json'), 'w') as outfile:
                json.dump(self.per_epoch_stats, outfile, indent=4)
                outfile.flush()
            with open(os.path.join(self.output_folder, 'summary.json'), 'w') as outfile:
                json.dump(self.summary, outfile, indent=4)
        self.output['hostname'] = socket.gethostname()
        with open(os.path.join(self.output_folder, f'{self.my_rank}_output.json'), 'w') as outfile:
            json.dump(self.output, outfile, indent=4)
            outfile.flush()
        if self.my_rank == 0:
            logging.info(f"{utcnow()} outputs saved in RANKID_output.json")


