

for model in clip-vit-base-patch32 clip-vit-base-patch16 clip-vit-large-patch14 imagebind flamingo9b; do
	echo -e "\n############## ${model} ###############\n"
	python cross_domain.py --real_train_path images/mmcelebahq_train \
				--real_test_path images/mmcelebahq_test \
				--fake_train_path images/DDIM_train \
				--fake_test_path images/DDIM_test \
				--model $model \
				--batch_size 64
done
