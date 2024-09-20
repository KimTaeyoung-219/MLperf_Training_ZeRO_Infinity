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

from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.common.enumerations import FormatType, FrameworkType, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.data_loader_factory import DataLoaderFactory
from dlio_benchmark.framework.framework import Framework, DummyTraceObject
from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
import os
import torch
import functools
import logging
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_profiler.logger import fn_interceptor as Profile

from time import sleep, time
from numpy import random

from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.storage.storage_factory import StorageFactory

HANDLED_FUNCTIONS = {}
dlp = Profile(MODULE_AI_FRAMEWORK)


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# Does this annotation mean that torch.mean will be replaced by torch_sleep?
@implements(torch.mean)
def torch_sleep(sleep_time):
    return sleep(sleep_time)


class TorchFramework(Framework):
    __instance = None

    @dlp.log_init
    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.PYTORCH
        super().init_loader(format_type, epoch, data_loader)

    @dlp.log
    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
        return TorchFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        pass

    @dlp.log
    def stop_framework_profiler(self):
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    @dlp.log
    def compute(self, x, epoch_number, step, computation_time):
        torch_sleep(computation_time)

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return False
    
    @dlp.log
    def compute_ZeRO_Infinity(self, batch, epoch, block_step, args, stats, block):
        computation_time = 0.0
        computation_time += self.forward(batch, epoch, block_step, args, stats, block)
        computation_time += self.backward(batch, epoch, block_step, args, stats, block)
        computation_time += self.step(batch, epoch, block_step, args, stats, block)
        return computation_time
        # torch_sleep(args.computation_time)
    
    @dlp.log
    def forward(self, batch, epoch, block_step, args, stats, block):
        # logging.info(f"Initiating FORWARD")
        total_time = 0.0
        for i in range(args.forward_epoch):
            # reading parameters from Nvme
            if args.parameter_swapping is True:
                batch, computation_time = self.load_tensor(args.parameter_num_files, args.forward_epoch, args.parameter_folder_list, i)
                logging.info(f"loading parameter: {batch.shape}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_parameter_read(epoch, block, computation_time)
            if args.forward_time_stdev > 0:
                forward_time = random.normal(args.forward_time_per_epoch, args.forward_time_stdev)
            else:
                forward_time = args.forward_time_per_epoch
            self.compute(batch, epoch, block_step, forward_time)
            total_time += forward_time
        return total_time
    
    @dlp.log
    def backward(self, batch, epoch, block_step, args, stats, block):
        # logging.info(f"Initiating BACKWARD")
        total_time = 0.0
        for i in range(args.backward_epoch):
            # reading parameters from Nvme
            if args.parameter_swapping is True:
                batch, computation_time = self.load_tensor(args.parameter_num_files, args.backward_epoch, args.parameter_folder_list, i)
                logging.info(f"loading parameter: {batch.shape}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_parameter_read(epoch, block, computation_time)
            if args.backward_time_stdev > 0:
                backward_time = random.normal(args.backward_time_per_epoch, args.backward_time_stdev)
            else:
                backward_time = args.backward_time_per_epoch
            self.compute(batch, epoch, block_step, backward_time)
            total_time += backward_time

            # writing gradients to Nvme
            if args.gradient_swapping is True:
                torch_size = args.gradient_size // args.gradient_num_files
                torch_dtype = args.gradient_dtype
                computation_time = self.save_tensor(args.gradient_num_files, args.backward_epoch, args.gradient_folder_list, i, torch_size, torch_dtype)
                logging.info(f"writing gradient: {torch_size}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_gradient_write(epoch, block, computation_time)
        return total_time
    
    @dlp.log
    def step(self, batch, epoch, block_step, args, stats, block):
        # logging.info(f"Initiating STEP")
        total_time = 0.0
        # logging.info(f"num_files: {args.optimizer_num_files}, epoch: {args.step_epoch}")
        for i in range(args.step_epoch):
            # reading optimizer from Nvme
            if args.optimizer_swapping is True:
                optim, computation_time = self.load_tensor(args.optimizer_num_files, args.step_epoch, args.optimizer_folder_list, i)
                logging.info(f"loading optimizer: {optim.shape}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_optimizer_read(epoch, block, computation_time)
            # reading gradient from Nvme
            if args.gradient_swapping is True:
                grad, computation_time = self.load_tensor(args.gradient_num_files, args.step_epoch, args.gradient_folder_list, i)
                logging.info(f"loading gradient: {grad.shape}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_gradient_read(epoch, block, computation_time)
            if args.step_time_stdev > 0:
                step_time = random.normal(args.step_time_per_epoch, args.step_time_stdev)
            else:
                step_time = args.step_time_per_epoch
            self.compute(batch, epoch, block_step, step_time)
            total_time += step_time

            # writing optimizer to Nvme
            if args.optimizer_swapping is True:
                torch_size = args.optimizer_size // args.optimizer_num_files
                torch_dtype = args.optimizer_dtype
                computation_time = self.save_tensor(args.optimizer_num_files, args.step_epoch, args.optimizer_folder_list, i, torch_size, torch_dtype)
                logging.info(f"writing optimizer: {torch_size}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_optimizer_write(epoch, block, computation_time)
            # writing parameter to Nvme
            if args.parameter_swapping is True:
                torch_size = args.parameter_size // args.parameter_num_files
                torch_dtype = args.parameter_dtype
                computation_time = self.save_tensor(args.parameter_num_files, args.step_epoch, args.parameter_folder_list, i, torch_size, torch_dtype)
                logging.info(f"writing parameter: {torch_size}, computation_time: {computation_time}")
                stats.ZeRO_Infinity_processed_parameter_write(epoch, block, computation_time)
        return total_time

    @dlp.log
    def find_file_range(self, epoch, num_files, max_epoch):
        step = (num_files // max_epoch) - 1
        if epoch < (num_files % max_epoch):
            step += 1
        start =  epoch * (num_files // max_epoch) + min(epoch, (num_files % max_epoch))
        end = start + step
        if start >= num_files:
            start, end = -1, -1
        # logging.info(f"start: {start}, end: {end}")
        return start, end
    
    @dlp.log
    def load_tensor(self, num_files, epoch, list, i):
        batch = torch.empty(0)
        start, end = self.find_file_range(i, num_files, epoch)
        # logging.info(f"start: {start}, end: {end}")
        t0 = time()
        if start != -1:
            for j in range(start, end + 1):
                file_path = list[j]
                loaded = torch.load(file_path)
                batch = torch.cat((batch, loaded), dim=0)
        # logging.info(batch.shape)
        t1 = time()
        computation_time = t1 - t0
        return batch, computation_time
    
    @dlp.log
    def save_tensor(self, num_files, epoch, list, i, torch_size, torch_dtype):
        random_tensor = torch.randn(torch_size, dtype=torch_dtype)
        start, end = self.find_file_range(i, num_files, epoch)
        t0 = time()
        if start != -1:
            for j in range(start, end + 1):
                file_path = list[j]
                torch.save(random_tensor, file_path)
        t1 = time()
        computation_time = t1 - t0
        return computation_time
