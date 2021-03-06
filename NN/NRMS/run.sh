CUDA_VISIBLE_DEVICES='1' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 10000 --train_steps 100000 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000'

CUDA_VISIBLE_DEVICES='1' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 50 --train_steps 100 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'nrms_title_test'

CUDA_VISIBLE_DEVICES='0' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 0 --train_steps 10000 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000' --load_checkpoint_path './checkpoint/nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000/checkpoint_steps_100000.cpt'

CUDA_VISIBLE_DEVICES='1' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 0 --train_steps 1 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000' --load_checkpoint_path './checkpoint/nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000/checkpoint_steps_100000.cpt'

# use abstract
CUDA_VISIBLE_DEVICES='0,1' python ../../main.py --use_abstract 1 --use_body 0 --news_max_len 128 --word_min_freq 30 --vocab_path '../../data/nrms_title_abstract_max_len_128_min_freq_30.vocab' --batch_size 128 --lr 0.0003 --warmup_steps 10000 --train_steps 500000 --steps_per_check 2000 --steps_per_checkpoint 100000 --experiment_name 'nrms_title_abs_max_len_128_min_freq_30_batch_128_lr_3e-4_warm_10000'

# title only
CUDA_VISIBLE_DEVICES='1' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 10000 --train_steps 300000 --steps_per_check 5000 --steps_per_checkpoint 50000 --experiment_name 'nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000_train_300000_dropout'

# submit
CUDA_VISIBLE_DEVICES='1' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 512 --lr 0.0003 --warmup_steps 0 --train_steps 1 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'prediction' --load_checkpoint_path './checkpoint/nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000_train_300000_without_dropout/best_model.cpt'

# 124 submit    
CUDA_VISIBLE_DEVICES='2' python ../../main.py --use_abstract 0 --use_body 0 --news_max_len 32 --word_min_freq 10 --vocab_path '../../data/nrms_title_max_len_32_min_freq_10.vocab' --batch_size 64 --lr 0.0003 --warmup_steps 0 --train_steps 1 --steps_per_check 2000 --steps_per_checkpoint 50000 --experiment_name 'prediction' --load_checkpoint_path './checkpoint/nrms_title_max_len_32_min_freq_10_batch_512_lr_3e-4_warm_10000_train_300000_without_dropout/best_model.cpt'