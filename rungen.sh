
for seed1 in 1 2 3 4 5
do
	python generators/gen_cont.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 2 --timeunites 10 --data cont --lr 0.001 --lambdao 10 --epochs 10 --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 2 --timeunites 10 --data cont --lr 0.001 --lambdao 0 --epochs 10 --seed $seed1  --seed2 $seed2
	done
done


for seed1 in 1 2 3 4 5
do
	python generators/gen_consecutive.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --timeunites 19 --data consecutive --pastunites 2 --lr 0.01 --lambdao 100 --epochs 20 --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --timeunites 19 --data consecutive --pastunites 2 --lr 0.01 --lambdao 0 --epochs 20 --seed $seed1 --seed2 $seed2
	done
done


for seed1 in 1 2 3 4 5
do
	python generators/gen_multieven.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multieven --lr 0.001 --lambdao 100 --epochs 10 --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multieven --lr 0.001 --lambdao 0 --epochs 10 --seed $seed1 --seed2 $seed2
	done
done

for seed1 in 1 2 3 4 5
do
	python generators/gen_multijump.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multijump --lr 0.001 --lambdao 100 --epochs 10 --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multijump --lr 0.001 --lambdao 0 --epochs 10 --seed $seed1 --seed2 $seed2
	done
done

for seed1 in 1 2 3 4 5
do
	python generators/gen_multistep.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multistep --lr 0.01 --lambdao 1 --epochs 10 --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --pastunites 5 --timeunites 19 --data multistep --lr 0.01 --lambdao 0 --epochs 10 --seed $seed1 --seed2 $seed2
	done
done

for seed1 in 1 2 3 4 5
do
	python generators/gen_unusual.py
	for seed2 in 1 2 3 4 5
	do
		python main.py --timeunites 19 --data unusual --pastunites 5 --lr 0.01 --lambdao 1000 --epochs 10  --seed $seed1 --seed2 $seed2
	done
	for seed2 in 1 2 3 4 5
	do
		python main.py --timeunites 19 --data unusual --pastunites 5 --lr 0.01 --lambdao 0 --epochs 10  --seed $seed1 --seed2 $seed2
	done
done