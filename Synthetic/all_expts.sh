# # Setting 1

mkdir setting1_30 setting1_20

for drop_prob in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
	python3 main.py --exp_name setting1_30/flat_$drop_prob --flat --epochs 30 --drop_prob $drop_prob &
	python3 main.py --exp_name setting1_30/only_label_$drop_prob --cascaded_step1 --epochs 20 --drop_prob $drop_prob
	python3 main.py --exp_name setting1_30/casc_$drop_prob --cascaded_step2 --epochs 30 --drop_prob $drop_prob --pretrained_label_model setting1_30/only_label_$drop_prob/19 &
	python3 main.py --exp_name setting1_30/jnt_$drop_prob --joint --epochs 30 --drop_prob $drop_prob &
done

for drop_prob in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
	python3 main.py --exp_name setting1_20/flat_$drop_prob --flat --epochs 30 --drop_prob $drop_prob &
	python3 main.py --exp_name setting1_20/only_label_$drop_prob --cascaded_step1 --epochs 20 --drop_prob $drop_prob
	python3 main.py --exp_name setting1_20/casc_$drop_prob --cascaded_step2 --epochs 30 --drop_prob $drop_prob --pretrained_label_model setting1_20/only_label_$drop_prob/19 &
	python3 main.py --exp_name setting1_20/jnt_$drop_prob --joint --epochs 30 --drop_prob $drop_prob &
done

# # 20%
# python3 main.py --exp_name flat_20 --flat --epochs 20 --drop_prob 0.2
# python3 main.py --exp_name only_label_20 --cascaded_step1 --epochs 20 --drop_prob 0.2
# python3 main.py --exp_name casc_20 --cascaded_step2 --epochs 20 --drop_prob 0.2 --pretrained_label_model only_label_20/19
# python3 main.py --exp_name jnt_20 --joint --epochs 20 --drop_prob 0.2

# # 40%
# python3 main.py --exp_name flat_40 --flat --epochs 50 --drop_prob 0.4
# python3 main.py --exp_name only_label_40 --cascaded_step1 --epochs 20 --drop_prob 0.4
# python3 main.py --exp_name casc_40 --cascaded_step2 --epochs 50 --drop_prob 0.4 --pretrained_label_model only_label_40/19
# python3 main.py --exp_name jnt_40 --joint --epochs 50 --drop_prob 0.4

# # 60%
# python3 main.py --exp_name flat_60 --flat --epochs 50 --drop_prob 0.6
# python3 main.py --exp_name only_label_60 --cascaded_step1 --epochs 20 --drop_prob 0.6
# python3 main.py --exp_name casc_60 --cascaded_step2 --epochs 50 --pretrained_label_model only_label_60/19
# python3 main.py --exp_name jnt_60 --joint --epochs 50 --drop_prob 0.6


# Setting 2

# for dataset_size in 2000 4000 8000 10000 16000 20000 30000; do
# 	echo $dataset_size
# 	python3 main.py --exp_name flat_$dataset_size --flat --epochs 30 --dataset_size $dataset_size
# 	python3 main.py --exp_name only_label_$dataset_size --cascaded_step1 --epochs 20 --dataset_size $dataset_size
# 	python3 main.py --exp_name casc_$dataset_size --cascaded_step2 --epochs 30 --pretrained_label_model only_label_$dataset_size/19 --dataset_size $dataset_size
# 	python3 main.py --exp_name jnt_$dataset_size --joint --epochs 30 --dataset_size $dataset_size
# done

# for dataset_size in 100 200 400 800 1000; do
# 	echo $dataset_size
# 	python3 main.py --exp_name flat_$dataset_size --flat --epochs 30 --dataset_size $dataset_size
# 	python3 main.py --exp_name only_label_$dataset_size --cascaded_step1 --epochs 20 --dataset_size $dataset_size
# 	python3 main.py --exp_name casc_$dataset_size --cascaded_step2 --epochs 30 --pretrained_label_model only_label_$dataset_size/19 --dataset_size $dataset_size
# 	python3 main.py --exp_name jnt_$dataset_size --joint --epochs 30 --dataset_size $dataset_size
# done
