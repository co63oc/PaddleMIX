# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import importlib

import paddle


def setup_dist(local_rank):
    if paddle.distributed.is_initialized():
        return
    paddle.device.set_device(device=local_rank)
    paddle.distributed.init_parallel_env()


def get_dist_info():
    if paddle.distributed.is_available():
        initialized = paddle.distributed.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = paddle.distributed.get_rank()
        world_size = paddle.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, help="module name", default="inference")
    parser.add_argument("--local_rank", type=int, nargs="?", help="for ddp", default=0)
    args, unknown = parser.parse_known_args()
    inference_api = importlib.import_module(args.module, package=None)
    inference_parser = inference_api.get_parser()
    inference_args, unknown = inference_parser.parse_known_args()
    paddle.seed(inference_args.seed)
    setup_dist(args.local_rank)
    # torch.backends.cudnn.benchmark = True
    rank, gpu_num = get_dist_info()
    print("@CoLVDM Inference [rank%d]: %s" % (rank, now))
    inference_api.run_inference(inference_args, gpu_num, rank)
