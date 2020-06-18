# Setting 1

mkdir setting1_30 setting1_20 setting2

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


# Setting 2

for dataset_size in 100 200 400 800 1000 2000 4000 8000 10000 16000 20000 30000; do
	echo $dataset_size
	python3 main.py --exp_name setting1/flat_$dataset_size --flat --epochs 30 --dataset_size $dataset_size
	python3 main.py --exp_name setting2/only_label_$dataset_size --cascaded_step1 --epochs 20 --dataset_size $dataset_size
	python3 main.py --exp_name setting2/casc_$dataset_size --cascaded_step2 --epochs 30 --pretrained_label_model setting2/only_label_$dataset_size/19 --dataset_size $dataset_size
	python3 main.py --exp_name setting2/jnt_$dataset_size --joint --epochs 30 --dataset_size $dataset_size
done
