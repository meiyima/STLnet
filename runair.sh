for l in 10 15 20 23
do
for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5
	do
		python main.py --timeunites $l --data airmulti --lr 0.1 --lambdao 100 --epochs 20 --seed $seed1 --seed2 $seed2
	done
done
done

for l in 10 15 20 23
do
for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5 
	do
		python main.py --timeunites $l --data airmulti --lr 0.1 --lambdao 0 --epochs 20 --seed $seed1 --seed2 $seed2
	done
done
done

echo noise >> record_res_airmulti.txt
for noisep in 0.02 0.04 0.06
do
echo $noisep >> record_res_airmulti.txt
for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5 
	do
		python main.py --timeunites 23 --data airmulti --lr 0.1 --lambdao 0 --epochs 20 --seed $seed1 --seed2 $seed2 --noisep $noisep
	done
done
echo $noisep >> record_res_airmulti.txt
for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5 
	do
		python main.py --timeunites 23 --data airmulti --lr 0.1 --lambdao 100 --epochs 20 --seed $seed1 --seed2 $seed2 --noisep $noisep
	done
done
done

echo DataPercentage >> record_res_airmulti.txt
for percentage in 0.8 0.85 0.9
do
echo $percentage >> record_res_airmulti.txt
for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5 
	do
		python main.py --timeunites 23 --data airmulti --percentage $percentage --lr 0.1 --lambdao 100 --epochs 20 --seed $seed1 --seed2 $seed2
	done
done

for seed1 in 1 2 3 4 5
do
	for seed2 in 1 2 3 4 5 
	do
		python main.py --timeunites 23 --data airmulti --percentage $percentage --lr 0.1 --lambdao 0 --epochs 20 --seed $seed1 --seed2 $seed2
	done
done
done