# result01
python run_bare.py --multirun preproc=False +tta=False train_dir=./input/1_200mg_train_211116_sugar_salt_wt%_10%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result02
python run_bare.py --multirun preproc=True +tta=True num_tta=10 train_dir=./input/1_200mg_train_211116_sugar_salt_wt%_10%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result03
model_file1=`python run.py --multirun num_tta=30 train_dir=./input/1_200mg_train_211116_sugar_salt_wt%_10%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/ | grep working_dir | cut -f2 -d' ' `
# result04
python predict.py --multirun num_tta=1,5,10,30,50 pretrain=${model_file1}/output/best_uptrain.model train_dir=./input/1_200mg_train_211116_sugar_salt_wt%_10%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result05
python run.py --multirun num_tta=30 train_dir=./input/2_50mg_211207_train10%_sugar_salt_wt%_random_100pics_square_spread_x10/ test_dir=./input/2_50mg_211207_test1%_sugar_salt_wt%_random_100pics_square_spread_x10/
# result06
python run.py --multirun num_tta=30 train_dir=./input/2_100mg_211207_train10%_sugar_salt_wt%_random_100pics_square_spread_x10/ test_dir=./input/2_100mg_211207_test1%_sugar_salt_wt%_random_100pics_square_spread_x10/
# result07
python run.py --multirun num_tta=30 train_dir=./input/4_200mg_train_211116_sugar_salt_wt%_20%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result08
python run.py --multirun num_tta=30 train_dir=./input/4_200mg_train_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result09
python run.py --multirun num_tta=30 train_dir=./input/train_dir1/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result10
python run.py --multirun num_tta=30 train_dir=./input/train_dir2/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result11
model_file2=`python run.py --multirun num_tta=30 train_dir=./input/train_dir3/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/ | grep working_dir | cut -f2 -d' ' `
# result12
python run.py --multirun num_tta=30 train_dir=./input/train_dir4/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result13
python run.py --multirun num_tta=30 train_dir=./input/train_dir5/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result14
python run.py --multirun num_tta=30 batch_size=8 no_validation=True train_dir=./input/5_sub2_random_10_v1/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result15
python run.py --multirun num_tta=30 batch_size=8 train_dir=./input/5_sub2_random_25_v1/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result16
python run.py --multirun num_tta=30 batch_size=8 train_dir=./input/5_sub2_random_50_v1/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result17
python run.py --multirun num_tta=30 train_dir=./input/6_train_alpha_gamma_glycine/ test_dir=./input/6_test_200mg_220120_alpha_gamma_glycine_wt%_random_100pics/
# result18
python run.py --multirun num_tta=30 flip=False train_dir=./input/9_train_D_L_tartaric_acid/ test_dir=./input/9_test_200mg_220329_D_L_tartaric_acid_wt%_random_100pics/
# result19
python run.py --multirun num_tta=30 train_dir=./input/10_train_220428_200mg_naphthalene/ test_dir=./input/10_test_220428_200mg_naphthalene_alumina_celite_silicagel_wt%_random_100pics/
# result20
python predict.py --multirun num_tta=30 pretrain=${model_file2}/output/best_uptrain.model train_dir=./input/7_A_train_ipod72ppi_200mg_sugar_salt_wt%_24pics/ test_dir=./input/7_A_test_ipod72ppi_200mg_sugar_salt_wt%_6pics/
# result21
python run.py --multirun num_tta=30 batch_size=8 warmup_epochs=50 no_validation=True pretrain=${model_file2}/output/best_uptrain.model train_dir=./input/7_A_train_ipod72ppi_200mg_sugar_salt_wt%_24pics/ test_dir=./input/7_A_test_ipod72ppi_200mg_sugar_salt_wt%_6pics/
# result22
python inference_speed.py --multirun num_tta=30 pretrain=${model_file2}/output/best_uptrain.model train_dir=./input/1_200mg_train_211116_sugar_salt_wt%_10%increment_random_100pics_square_x10/ test_dir=./input/1_200mg_test_211116_sugar_salt_wt%_1%increment_random_100pics_square_x10/
# result 23
python run.py --multirun num_tta=30 train_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_training25/ test_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_test10/
# result 24
python run.py --multirun num_tta=30 train_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_training50/ test_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_test10/
# result 25
model_file3=`python run.py --multirun num_tta=30 train_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_training75/ test_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_test10/ | grep working_dir | cut -f2 -d' ' `
train_dir=./input/1_Heating_Reaction_train_25_50_75/MAP_PAS_110_training75/ 
# result 26
python predict.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/2_Heating_Reaction_110C_1_6h_5pics/test_110C_5pics_1_6h/
# result 27
python predict.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/2_Heating_Reaction_110C_1_6h_5pics/test2_110C_6pics/
# result 28
python predict_only.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/3_1_Heating_Reaction_110C_rt_13pics/ +outdim=2
# result 29
python predict_only.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/3_2_Heating_Reaction_110_rt_63pics/ +outdim=2
# result 30
python predict_only.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/3_3_Heating_Reaction_110_115_93pics/ +outdim=2
# result 31
python predict.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/4_various_reaction_temp_80C_100C/test1_100C_5pics_72h/
# result 32
python predict.py --multirun num_tta=30 pretrain=${model_file3}/output/best_uptrain.model train_dir=${train_dir} test_dir=./input/4_various_reaction_temp_80C_100C/test2_80C_5pics_120h/
