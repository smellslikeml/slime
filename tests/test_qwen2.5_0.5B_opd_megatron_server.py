import json
import os
import shlex
import subprocess
import time
import urllib.request

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4

TEACHER_HOST = "127.0.0.1"
TEACHER_PORT = 13142
TEACHER_WARMUP_PORT = 13143


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _ray_runtime_env_json():
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    return json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "RAY_USE_UVLOOP": "0",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0"),
                "no_proxy": f"127.0.0.1,{master_addr}",
                "MASTER_ADDR": master_addr,
            }
        }
    )


def _teacher_train_args():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--max-tokens-per-gpu 4096 "
    )

    misc_args = (
        "--debug-train-only "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--megatron-to-hf-mode bridge "
    )

    server_args = (
        f"--teacher-port {TEACHER_PORT} "
        f"--teacher-warmup-port {TEACHER_WARMUP_PORT} "
        "--teacher-warmup-timeout-s 3000 "
    )

    return f"{ckpt_args} {perf_args} {misc_args} {server_args}"


def _launch_teacher_server():
    log_path = "/tmp/megatron_teacher_server.log"
    submission_id = f"megatron_teacher_{int(time.time())}_{os.getpid()}"
    entrypoint = (
        f"exec > {log_path} 2>&1 && "
        "set -euxo pipefail && "
        f"source {U.repo_base_dir}/scripts/models/{MODEL_TYPE}.sh && "
        "export PYTHONUNBUFFERED=1 && "
        "python3 -m slime.backends.megatron_utils.server.megatron_server "
        "${MODEL_ARGS[@]} "
        f"{_teacher_train_args()} "
    )
    cmd = (
        "ray job submit "
        "--address=http://127.0.0.1:8265 "
        f"--submission-id {submission_id} "
        "--no-wait "
        "--entrypoint-num-cpus 0 "
        f"--runtime-env-json={shlex.quote(_ray_runtime_env_json())} "
        f"-- bash -lc {shlex.quote(entrypoint)}"
    )

    result = subprocess.run(["bash", "-lc", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit Megatron teacher server job:\n{result.stdout}")

    print(f"Starting Megatron teacher server job {submission_id}, log: {log_path}")
    for _ in range(180):
        status = _get_teacher_job_status(submission_id)
        if any(terminal in status for terminal in ("FAILED", "STOPPED", "SUCCEEDED")):
            _dump_teacher_job_debug(submission_id, log_path)
            raise RuntimeError(f"Megatron teacher server job exited before serving: {status}")

        try:
            req = urllib.request.urlopen(f"http://{TEACHER_HOST}:{TEACHER_PORT}/healthz", timeout=2)
            if req.status == 200:
                print(f"Megatron teacher server is ready at {TEACHER_HOST}:{TEACHER_PORT}")
                return submission_id
        except Exception:
            pass
        time.sleep(5)

    _dump_teacher_job_debug(submission_id, log_path)
    raise RuntimeError(f"Megatron teacher server failed to start within timeout. Check {log_path}")


def _get_teacher_job_status(submission_id: str) -> str:
    result = subprocess.run(
        ["ray", "job", "status", "--address=http://127.0.0.1:8265", submission_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def _dump_teacher_job_debug(submission_id: str, log_path: str):
    U.exec_command(f"ray job status --address=http://127.0.0.1:8265 {submission_id}; true")
    U.exec_command(f"ray job logs --address=http://127.0.0.1:8265 {submission_id} | tail -200; true")
    U.exec_command(f"tail -200 {log_path}; true")


def _stop_teacher_server(submission_id: str | None):
    if submission_id:
        U.exec_command(f"ray job stop --address=http://127.0.0.1:8265 --no-wait {submission_id}; true")
    U.exec_command("pkill -f slime.backends.megatron_utils.server.megatron_server; true")


def execute():
    teacher_submission_id = None

    def launch_teacher():
        nonlocal teacher_submission_id
        teacher_submission_id = _launch_teacher_server()

    try:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
        )

        eval_args = (
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        rm_args = (
            "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
            "--custom-reward-post-process-path "
            "slime.rollout._opd_test_helpers.post_process_megatron_server_opd_rewards "
            f"--rm-url http://{TEACHER_HOST}:{TEACHER_PORT}/generate "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            "--use-opd "
            "--opd-type sglang "
            "--opd-kl-coef 1.0 "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.00 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )

        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
        )

        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-cuda-graph-max-bs 16 "
            "--sglang-enable-metrics "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
            "--colocate "
            "--megatron-to-hf-mode bridge "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{eval_args} "
            f"{sglang_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{rm_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_teacher,
        )
    finally:
        _stop_teacher_server(teacher_submission_id)
        U.exec_command("pkill -9 sglang; true")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
