

for model in clip-vit-base-patch16 imagebind flamingo9b; do
	echo -e "\n############## ${model} ###############\n"
	python cross_domain.py --real_train_path images/mmcelebahq_train \
				--real_test_path ../diffusion_face_split/test/mmcelebahq \
				--fake_train_path images/DDIM_train \
				--fake_test_path ../diffusion_face_split/test/SD_2.1_I2I/ \
				--model $model \
				--batch_size 2
done
