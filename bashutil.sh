function dist-train(){
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
    export NCCL_IB_GID_INDEX=3
    export NCCL_SOCKET_IFNAME=eth0
    python3 -m torch.distributed.launch \
        --nproc_per_node=$ARNOLD_WORKER_GPU \
        --nnodes=$ARNOLD_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=$ARNOLD_WORKER_0_HOST \
        --master_port=$ARNOLD_WORKER_0_PORT \
        fairseq_cli/train.py \
        "$@"
}

function dist-train-multi-transformer(){
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
    export NCCL_IB_GID_INDEX=3
    export NCCL_SOCKET_IFNAME=eth0
    python3 -m torch.distributed.launch \
        --nproc_per_node=$ARNOLD_WORKER_GPU \
        --nnodes=$ARNOLD_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=$ARNOLD_WORKER_0_HOST \
        --master_port=$ARNOLD_WORKER_0_PORT \
        fairseq_cli/train_multi_trans.py \
        "$@"
}

function dist-train-mglat(){
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
    export NCCL_IB_GID_INDEX=3
    export NCCL_SOCKET_IFNAME=eth0
    python3 -m torch.distributed.launch \
        --nproc_per_node=$ARNOLD_WORKER_GPU \
        --nnodes=$ARNOLD_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=$ARNOLD_WORKER_0_HOST \
        --master_port=$ARNOLD_WORKER_0_PORT \
        fairseq_cli/train_mglat.py \
        "$@"
}

function dist-train-mglat-diffusion(){
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
    export NCCL_IB_GID_INDEX=3
    export NCCL_SOCKET_IFNAME=eth0
    python3 -m torch.distributed.launch \
        --nproc_per_node=$ARNOLD_WORKER_GPU \
        --nnodes=$ARNOLD_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=$ARNOLD_WORKER_0_HOST \
        --master_port=$ARNOLD_WORKER_0_PORT \
        fairseq_cli/train_mglat_diffusion.py \
        "$@"
}

function debug-train(){
    CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen localhost:5678 fairseq_cli/train.py "$@"
}

install_dependencies(){
    export http_proxy=http://sys-proxy-rd-relay.byted.org:3128
    export https_proxy=http://sys-proxy-rd-relay.byted.org:3128

    pip3 install Cython
    python3 setup.py build_ext --inplace
    python3 setup.py develop --user

    pip3 install sacremoses
}


auto_install_dependencies(){
    if [[ "$ARNOLD_DEBUG" = "vscode" ]]
    then
        echo 'in debug mode, some steps are skipped' >&2
    else
        install_dependencies
    fi
}

function parse_args(){
    while [[ "$#" -gt 0 ]]; do
        found=0
        for key in "${!BASH_ARGS[@]}"; do
            if [[ "--$key" == "$1" ]] ; then
                BASH_ARGS[$key]=$2
                found=1
            fi
        done
        if [[ $found == 0 ]]; then
            echo "arg $1 not defined!" >&2
            exit 1
        fi
        shift; shift
    done

    echo "======== PARSED BASH ARGS ========" >&2
    for key in "${!BASH_ARGS[@]}"; do
        echo "    $key = ${BASH_ARGS[$key]}" >&2
        eval "$key=${BASH_ARGS[$key]}" >&2
    done
    echo "==================================" >&2
}
