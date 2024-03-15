export NCCL_DEBUG=WARN

best_or_last=best
enc_langtok=src
subset=test
capacity_factor=0.5
n_process=8
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--save_dir)
            save_dir=$2
            shift
            shift
            ;;
        -n|--n_process)
            n_process=$2
            shift
            shift
            ;;
        -s|--subset)
            subset=$2
            shift
            shift
            ;;
        -l|--last)
            best_or_last=last
            shift
            ;;
        -c|--capacity_factor )
            capacity_factor=$2
            shift
            shift
            ;;
        -*|--*)
            echo "unkown option $1"
            exit 1
            ;;
    esac
done
# save
echo save_dir=$save_dir
echo subset=$subset
echo best_or_last=$best_or_last

# save_dir="/data12/checkpoints/opus-100/ddp/base_srctok/"
model_name=(${save_dir//// })
model_name=${model_name[-1]}
model_name=${model_name}${prefix}
translation_dir=translation_data/$model_name
score_path=bleu/$model_name.bleu

echo "model_name:${model_name}"
echo "prefix:${prefix}"
echo "translation_dir:${translation_dir}"
echo "score_path:${score_path}"

#distributed
master_addr="127.0.0.3"
master_port=12345

# data
root_data_dir=./mmt_data/opus-100-preprocessed
main_data_bin_dir=${root_data_dir}/main_data_bin
extra_data_bin_dir=${root_data_dir}/extra_data_bin

spm_data_dir=${root_data_dir}/spm_data
spm_corpus_dir=${root_data_dir}/spm_corpus

max_tokens=6000

all_lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"
lang_dict=${root_data_dir}/lang_dict.txt

python=python
sacrebleu=sacrebleu

echo "spm decode complete!"
checkpoint='checkpoint_best'
checkpoint_path="${save_dir}/${checkpoint}.pt"

mkdir -p ${translation_dir}

lang_pairs=${all_lang_pairs//,/ }
result_path=${translation_dir}
echo "write translation to ${translation_dir}"

moe_args="
    --ddp-backend fully_sharded \
    --is-moe"

# export CUDA_VISIBLE_DEVICES="5,6"
# for generate_multiple.py, --source-lang and --target-lang does not work, it would iterate all languages in lang-pairs-to-generate
${python} generate_multiple.py ${main_data_bin_dir} \
--task translation_multi_simple_epoch \
--user-dir ./hmoe \
--distributed-world-size ${n_process} \
--lang-pairs ${all_lang_pairs} \
--lang-dict ${lang_dict} \
--source-dict ${main_data_bin_dir}/dict.txt \
--target-dict ${main_data_bin_dir}/dict.txt \
--decoder-langtok \
--encoder-langtok src \
--enable-lang-ids \
--source-lang en \
--target-lang eu \
--gen-subset ${subset} \
--path ${checkpoint_path} \
--max-tokens ${max_tokens} \
--beam 5 \
--results-path ${result_path} \
--post-process sentencepiece \
--lang-pairs-to-generate $lang_pairs \
--skip-invalid-size-inputs-valid-test \
--model-overrides "{'moe_eval_capacity_token_fraction':${capacity_factor}}" \
--ddp-backend fully_sharded \
 ${moe_args}

for lang_pair in ${lang_pairs// / }; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}

    parallel_trans_dir=${translation_dir}/${lang_pair}
    echo "compute bleu for ${lang_pair}"
    ${python} -u ./translation_utils/extract_translation.py \
        --translation_file_path ${parallel_trans_dir}/generate-${subset}.txt \
        --output_hp_file_path ${parallel_trans_dir}/extract.${subset}.txt \
        --output_ref_file_path ${parallel_trans_dir}/gt.${subset}.txt \
    
    score=$(${sacrebleu} -l ${lang_pair} -w 6 ${parallel_trans_dir}/gt.${subset}.txt < ${parallel_trans_dir}/extract.${subset}.txt)
    
    score=$(echo $score | grep -Po ":\s(\d+\.*\d*)" | head -n 1 | grep -Po "\d+\.*\d*")
    
    echo "${lang_pair}: ${score}" >> ${score_path}
done
