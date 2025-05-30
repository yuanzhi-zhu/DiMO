#!/bin/bash


####################################################################################################
### generate samples for evaluation
####################################################################################################



####################################################################################################
### MaskGit
####################################################################################################

export MASTER_PORT=29501
DID=0,1,2,3,4,5,6,7
DID=0,1,2,3
# DID=0,1,2
# DID=4,5,6,7
# DID=0
# DID=0,1
IFS=',' read -r -a array <<< "$DID"
num_gpu_per_node=${#array[@]}

ref_dir_list=(
    ./codes/Maskgit_fast/outputs_/MaskGit_discrete_benchmark/DDMD-GPUS2-bs64-grad_accu1-glr1e-05-flr1e-05-FKL-fix_rfix_0.6-emb_pert_fix_0.1-tcfg2.0-fcfg1.0-fcfgt1.0-r_mode_arccos-r_mode_f_arccos-reduce_sum-seed3407-bf16/2025-03-14T22-11
)

# result_dir=./codes/Maskgit_fast/outputs_MaskGit_discrete
# # get all the directories in the result_dir
# ref_dir_list=($(ls -d $result_dir/*))
# # get all the subdirectories in the ref_dir_list
# ref_dir_list=($(ls -d $result_dir/*/*/))

ckpt_list=(
    # checkpoint-8000
    # _checkpoint-5000
    # _checkpoint-10000
    # _checkpoint-15000
    # _checkpoint-20000
    # _checkpoint-25000
    _checkpoint-30000
    # _checkpoint-35000
    # checkpoint-38000
    # _checkpoint-40000
    # checkpoint-42000
)


# ckpt_ref_path=${ref_dir_list[0]}/${ckpt_list[0]}
for i in "${!ref_dir_list[@]}"; do
    for j in "${!ckpt_list[@]}"; do
        # for gen_temp in 1. 3. 4. 5. 7.; do
        # for gen_temp in 4.5 6. 7. 8.; do
        # for gen_temp in 8. 9. 10 ; do
        for gen_temp in 1. ; do
            network_dir=${ref_dir_list[$i]}/${ckpt_list[$j]}/ema_model
            # network_dir=${ref_dir_list[$i]}/${ckpt_list[$j]}/generator

            # if network_dir is not a directory, skip
            if [ ! -d $network_dir ]; then
                continue
            fi

            # CUDA_VISIBLE_DEVICES=$DID python3 sample_MaskGit.py \
            CUDA_VISIBLE_DEVICES=$DID torchrun --nproc_per_node=$num_gpu_per_node sample_MaskGit.py \
                --save_dir ./samples_test/ema2_50000 \
                --vit_path $network_dir \
                --mode generate \
                --gen_temp $gen_temp \
                --nb_sample 50000

            # echo ./samples_test/$outdir_name
            CUDA_VISIBLE_DEVICES=$DID python3 evaluation/fid.py \
                --fdir2 ./samples_test/ema2_50000 \
                --ref_stat ./fid_stats_imagenet256_guided_diffusion.npz \
                --network_dir $network_dir \
                --gen_temp $gen_temp \
                --log_file fid_eval.log

            echo $network_dir
            
        done
    done
done

            
# # ./codes/Maskgit_fast/MaskGit_src/samples_50000_2
# # PRDC
# # for fdir2 in samples_50000_2 samples_50000_4 samples_50000_8 samples_50000_16 samples_50000_32
# for fdir in samples_50000_8
# do
#     python3 prdc.py \
#         --fdir ./codes/Maskgit_fast/MaskGit_src/$fdir 
# done


####################################################################################################
### Meissonic
####################################################################################################


ref_dir_list=(
    # ./codes/Maskgit_fast/outputs_MaskGit_discrete_FKL_init2/DDMD-GPUS1-bs32-grad_accu2-glr1e-05-flr1e-05-FKL-fix_r0.7-emb_pert0.0-tcfg2.0-fcfg1.0-fcfgt1.0-r_mode_arccos-r_mode_f_arccos-a_fake0.0-reduce_sum-seed42-ema0.9999-fp32/2025-02-16T09-53
    # ./codes/Maskgit_fast/outputs_MaskGit_discrete/DDMD-GPUS1-bs32-grad_accu2-glr1e-05-flr1e-05-Jeffreys_0.-fix_r0.5-emb_pert0.3-tcfg2.0-fcfg1.0-fcfgt1.0-r_mode_arccos-r_mode_f_arccos-a_fake0.0-reduce_sum-seed42-ema0.9999-fp32/2025-02-15T06-26
    ./codes/Maskgit_fast/outputs_Meissonic_discrete/DDMD-GPUS1-bs2-grad_accu8-glr1e-06-flr1e-06-FKL-fix_r0.5-emb_pert_fix_0.3-tcfg4.0-fcfg1.0-fcfgt1.0-r_mode_cosine-r_mode_f_cosine-a_fake0.0-reduce_sum-seed42-ema0.9999-bf16/2025-02-19T16-00
)

# result_dir=./codes/Maskgit_fast/outputs_MaskGit_discrete
# # get all the directories in the result_dir
# ref_dir_list=($(ls -d $result_dir/*))
# # get all the subdirectories in the ref_dir_list
# ref_dir_list=($(ls -d $result_dir/*/*/))

ckpt_list=(
    # _checkpoint-5000
    # _checkpoint-10000
    # _checkpoint-15000
    # _checkpoint-20000
    # _checkpoint-25000
    checkpoint-13000
)


# ckpt_ref_path=${ref_dir_list[0]}/${ckpt_list[0]}
for i in "${!ref_dir_list[@]}"; do
    for j in "${!ckpt_list[@]}"; do
        # for gen_temp in 1. 3. 4. 5. 7.; do
        # for gen_temp in 4.5 6. 7.; do
        for gen_temp in 0.5 ; do
            network_dir=${ref_dir_list[$i]}/${ckpt_list[$j]}/ema_model

            # if network_dir is not a directory, skip
            if [ ! -d $network_dir ]; then
                continue
            fi
            
            network_dir='./models/meissonic'
            CUDA_VISIBLE_DEVICES=0 torchrun \
                    --nproc_per_node=$num_gpu_per_node \
                    --master_port=29501 \
                sample_Meissonic.py \
                --save_dir ./samples/Meissonic_ema_50000 \
                --vit_path $network_dir \
                --gen_temp $gen_temp \
                --mode generate_FID \

            # # echo ./samples_test/$outdir_name
            # CUDA_VISIBLE_DEVICES=$DID python3 fid.py \
            #     --fdir2 ./samples_test/Meissonic_ema_50000 \
            #     --ref_stat ./fid_stats_imagenet256_guided_diffusion.npz \
            #     --network_dir $network_dir \
            #     --gen_temp $gen_temp \
            #     --log_file fid_eval_Meissonic.log

            # CUDA_VISIBLE_DEVICES=$DID python3 gigagan_eval_clip.py \
            #         --ref_dir ./samples_test/Meissonic_ema_50000 \
            #         --caption_file ./prompts/captions.txt \
            # echo $network_dir
        

            # CUDA_VISIBLE_DEVICES=$DID torchrun \
            #         --nproc_per_node=$num_gpu_per_node \
            #         --master_port=29501 \
            #     sample_Meissonic.py \
            #     --save_dir ./samples_test/Meissonic_ema_50000_GenEval \
            #     --vit_path $network_dir \
            #     --mode generate_GenEval \
            #     --gen_temp $gen_temp \

            # export PYTHONPATH=$PYTHONPATH:./:./geneval
            # python geneval/evaluation/evaluate_images.py \
            #     "./samples_test/Meissonic_ema_50000_GenEval" \
            #     --outfile "samples_test/GenEval_results.jsonl" \
            #     --model-path "samples_test/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"


            # CUDA_VISIBLE_DEVICES=$DID python3 generate_onestep_Meissonic.py \
            #     --save_dir ./samples_test/test \
            #     --vit_path $network_dir \
            #     --mode sample \
            
        done
    done
done

